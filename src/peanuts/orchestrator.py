"""Async multi-agent orchestrator, worktree manager, and Single-Writer Integrator."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from peanuts.config import PeanutsConfig
from peanuts.db import PeanutsRepository
from peanuts.git_ops import GitError, WorktreeManager, detect_git_context
from peanuts.models import (
    AgentRecord,
    AgentStatus,
    AskHumanAction,
    DelegateAction,
    DelegationRecord,
    EventRecord,
    IntegrateAction,
    MergeJobRecord,
    MergeJobStatus,
    MessageRecord,
    ProviderName,
    QuestionRecord,
    ReplyAction,
    SystemStateResponse,
)
from peanuts.protocol import build_agent_prompt, parse_agent_action
from peanuts.runners import (
    AgentRunner,
    ClaudeRunner,
    CodexRunner,
    MockRunner,
    RunnerError,
    TurnInput,
)


class EventBus:
    """In-process async fan-out bus for persisted EventRecord notifications."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[EventRecord]] = set()

    def publish(self, events: Sequence[EventRecord]) -> None:
        for ev in events:
            for queue in list(self._subscribers):
                try:
                    queue.put_nowait(ev)
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()
                        queue.put_nowait(ev)
                    except asyncio.QueueEmpty:
                        pass

    @asynccontextmanager
    async def subscribe(self, maxsize: int = 512) -> AsyncIterator[asyncio.Queue[EventRecord]]:
        queue: asyncio.Queue[EventRecord] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)


class Orchestrator:
    """Coordinates the root mediator, isolated worktree child workers, and single Integrator."""

    def __init__(
        self,
        config: PeanutsConfig,
        *,
        runners: Mapping[ProviderName, AgentRunner] | None = None,
    ) -> None:
        self.config = config
        self.repo = PeanutsRepository(config.db_path)
        self.event_bus = EventBus()
        self._runners: dict[ProviderName, AgentRunner] = (
            dict(runners)
            if runners is not None
            else {
                "claude": ClaudeRunner(binary=config.claude_bin),
                "codex": CodexRunner(binary=config.codex_bin),
                "mock": MockRunner(),
            }
        )
        self._tasks: set[asyncio.Task[None]] = set()
        self._merge_task: asyncio.Task[None] | None = None
        self._closed = False
        self._quiescent_event = asyncio.Event()
        self._quiescent_event.set()
        self._pump_lock: asyncio.Lock | None = None

        # Detect git repo & base branch (the branch where peanuts was launched = parent's branch)
        self.git_ctx = detect_git_context(config.cwd)
        self.worktree_mgr: WorktreeManager | None = (
            WorktreeManager(self.git_ctx)
            if (config.enable_worktrees and self.git_ctx.supports_worktrees)
            else None
        )

        resolved_root_provider = config.resolve_root_provider()
        root, root_events = self.repo.ensure_root_agent(
            resolved_root_provider,
            worktree_path=str(self.worktree_mgr.repo_root) if self.worktree_mgr else None,
            branch_name=self.git_ctx.base_branch,
        )
        self.root_agent_id = root.id

        extra_events: list[EventRecord] = []
        extra_events.extend(self.repo.recover_interrupted_turns())
        extra_events.extend(self.repo.recover_interrupted_merge_jobs())
        if self.worktree_mgr is not None:
            self.worktree_mgr.abort_merge()
            integrator, int_events = self.repo.ensure_integrator_agent(
                config.resolve_child_provider(),
                worktree_path=str(self.worktree_mgr.repo_root),
                branch_name=self.worktree_mgr.base_branch,
            )
            extra_events.extend(int_events)
            if config.root_provider is not None and integrator.provider != config.resolve_child_provider():
                _, int_ov = self.repo.update_agent_provider(integrator.id, config.resolve_child_provider())
                extra_events.extend(int_ov)

        if config.root_provider is not None and root.provider != config.root_provider:
            _, override_events = self.repo.update_agent_provider(root.id, config.root_provider)
            extra_events.extend(override_events)

        self.event_bus.publish(root_events + extra_events)

    def _get_pump_lock(self) -> asyncio.Lock:
        if self._pump_lock is None:
            self._pump_lock = asyncio.Lock()
        return self._pump_lock

    def available_providers(self) -> list[ProviderName]:
        if self.config.root_provider == "mock" and self.config.default_child_provider in (None, "mock"):
            return ["mock"]
        real_available: list[ProviderName] = [
            item.name
            for item in self.config.detect_providers()
            if item.available and item.name != "mock" and item.name in self._runners
        ]
        if real_available:
            return real_available
        return ["mock"]

    def get_state(self, message_limit: int = 100) -> SystemStateResponse:
        return SystemStateResponse(
            root_agent_id=self.root_agent_id,
            base_branch=self.git_ctx.base_branch,
            worktrees_enabled=self.worktree_mgr is not None,
            latest_seq=self.repo.latest_seq(),
            agents=self.repo.list_agents(),
            delegations=self.repo.list_delegations(),
            open_questions=self.repo.list_questions(only_open=True),
            merge_jobs=self.repo.list_merge_jobs(),
            recent_messages=self.repo.list_messages(limit=message_limit),
        )

    async def shutdown(self) -> None:
        """Cancel any active background tasks, ensure clean git state, and close SQLite."""
        self._closed = True
        tasks = list(self._tasks)
        if self._merge_task is not None and not self._merge_task.done():
            tasks.append(self._merge_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._merge_task = None
        if self.worktree_mgr is not None:
            self.worktree_mgr.abort_merge()
        self.repo.close()

    # ------------------------------------------------------------------
    # Public Commands
    # ------------------------------------------------------------------

    async def send_human_message(
        self,
        content: str,
        *,
        target_agent_id: str | None = None,
        theme: str | None = None,
        provider: ProviderName | None = None,
    ) -> tuple[str, MessageRecord]:
        agent_id = target_agent_id or self.root_agent_id
        if provider is not None:
            _, prov_events = self.repo.update_agent_provider(agent_id, provider)
            self.event_bus.publish(prov_events)

        msg, events = self.repo.enqueue_human_message(agent_id, content, theme=theme)
        self.event_bus.publish(events)
        await self.pump()
        return agent_id, msg

    async def answer_question(self, question_id: str, answer: str) -> QuestionRecord:
        agent, question, events = self.repo.answer_question(question_id, answer)
        self.event_bus.publish(events)
        # If the question belonged to the Integrator waiting on a dirty base branch,
        # consume the dummy inbox item and re-queue waiting merge jobs directly.
        if agent.role_kind == "integrator":
            started = self.repo.start_inbox_turn(agent.id)
            if started is not None:
                _, _, start_evs = started
                self.event_bus.publish(start_evs)
                idle_evs = self.repo.set_integrator_status(AgentStatus.IDLE)
                self.event_bus.publish(idle_evs)
            resumed_evs = self.repo.resume_waiting_merge_jobs()
            self.event_bus.publish(resumed_evs)
        await self.pump()
        return question

    async def retry_agent(self, agent_id: str) -> AgentRecord:
        agent, events = self.repo.reset_failed(agent_id)
        self.event_bus.publish(events)
        await self.pump()
        return agent

    async def cancel_delegation(self, delegation_id: str) -> DelegationRecord:
        delegation, events = self.repo.cancel_delegation(delegation_id)
        self.event_bus.publish(events)
        await self.pump()
        return delegation

    async def set_agent_provider(self, agent_id: str, provider: ProviderName) -> AgentRecord:
        agent, events = self.repo.update_agent_provider(agent_id, provider)
        self.event_bus.publish(events)
        return agent

    async def enqueue_merge_for_agent(self, agent_id: str) -> MergeJobRecord:
        """Manually commit & enqueue a worker agent's worktree branch to the Integrator queue."""
        if self.worktree_mgr is None:
            raise GitError("Git worktree isolation is not active in this workspace")
        agent = self.repo.get_agent(agent_id)
        if agent is None:
            raise KeyError(f"Agent not found: {agent_id}")
        if not agent.branch_name or not agent.worktree_path or agent.role_kind != "worker":
            raise GitError(f"Agent {agent_id} does not own a mergeable worker branch")
        self.worktree_mgr.auto_commit_worktree(
            Path(agent.worktree_path),
            f"peanuts({agent.theme}): manual merge commit",
        )
        job, events = self.repo.enqueue_merge_job(
            worker_agent_id=agent.id,
            theme=agent.theme,
            source_branch=agent.branch_name,
            target_branch=self.worktree_mgr.base_branch,
        )
        self.event_bus.publish(events)
        await self.pump()
        return job

    async def resume_merge_queue(self) -> list[MergeJobRecord]:
        """Re-queue any merge jobs paused on WAITING_CLEAN_BASE and trigger Integrator."""
        events = self.repo.resume_waiting_merge_jobs()
        self.event_bus.publish(events)
        await self.pump()
        return self.repo.list_merge_jobs()

    # ------------------------------------------------------------------
    # Scheduler Pump
    # ------------------------------------------------------------------

    async def pump(self) -> None:
        """Advance all ready agent turns and Single-Writer Integrator merge jobs."""
        if self._closed:
            return
        async with self._get_pump_lock():
            if self._closed:
                return
            # 1. Wake non-root children waiting on grandchildren
            for parent in self.repo.list_waiting_parents_ready_to_resume():
                resumed = self.repo.resume_parent_after_children(parent.id)
                if resumed is not None:
                    _, events = resumed
                    self.event_bus.publish(events)

            # 2. Start PENDING delegations whose callee is IDLE (provisioning worktree if enabled)
            for delegation in self.repo.list_pending_delegations():
                started = self.repo.start_delegation(delegation.id)
                if started is not None:
                    callee, active_del, events = started
                    callee = self._ensure_worker_worktree(callee)
                    self.event_bus.publish(events)
                    self._spawn_turn(callee, active_del.prompt)

            # 3. Start IDLE agents with queued inbox messages (skipping integrator which is driven by merge queue)
            for idle_agent in self.repo.list_idle_agents_with_inbox():
                if idle_agent.role_kind == "integrator":
                    continue
                started_turn = self.repo.start_inbox_turn(idle_agent.id)
                if started_turn is not None:
                    running_agent, inbox_item, events = started_turn
                    running_agent = self._ensure_worker_worktree(running_agent)
                    self.event_bus.publish(events)
                    self._spawn_turn(running_agent, inbox_item.content)

            # 4. Trigger Single-Writer Integrator if a QUEUED merge job exists and none is running
            if (
                self.worktree_mgr is not None
                and (self._merge_task is None or self._merge_task.done())
                and not self.repo.has_active_merge_job()
            ):
                next_job = self.repo.claim_next_queued_merge_job()
                if next_job is not None:
                    self._quiescent_event.clear()
                    self._merge_task = asyncio.create_task(
                        self._run_merge_job(next_job),
                        name=f"peanuts-merge-{next_job.id}",
                    )
                    self._merge_task.add_done_callback(self._on_merge_done)

            self._update_quiescent_flag()

    def _ensure_worker_worktree(self, agent: AgentRecord) -> AgentRecord:
        if self.worktree_mgr is None or agent.role_kind != "worker":
            return agent
        if agent.worktree_path is not None and agent.branch_name is not None and Path(agent.worktree_path).exists():
            return agent
        wt_path, branch_name = self.worktree_mgr.create_worktree(agent.id, agent.theme)
        return self.repo.set_agent_worktree(agent.id, str(wt_path), branch_name)

    def _update_quiescent_flag(self) -> None:
        merge_running = self._merge_task is not None and not self._merge_task.done()
        if self._tasks or merge_running:
            self._quiescent_event.clear()
        else:
            self._quiescent_event.set()

    def _spawn_turn(self, agent: AgentRecord, user_input: str) -> None:
        self._quiescent_event.clear()
        task = asyncio.create_task(self._execute_turn(agent, user_input), name=f"peanuts-turn-{agent.id}")
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        self._update_quiescent_flag()

    def _on_merge_done(self, _task: asyncio.Task[None]) -> None:
        self._merge_task = None
        self._update_quiescent_flag()
        if not self._closed:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.pump())
            except RuntimeError:
                pass

    async def wait_until_quiescent(self, timeout: float = 15.0) -> None:
        """Wait until all currently running turns and merge jobs settle."""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            await self.pump()
            merge_running = self._merge_task is not None and not self._merge_task.done()
            if not self._tasks and not merge_running:
                return
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for orchestrator to become quiescent")
            await asyncio.wait_for(self._quiescent_event.wait(), timeout=remaining)

    def _turn_permissions(self, agent: AgentRecord) -> tuple[str | None, str | None]:
        """Parent (root) runs read-only; workers & integrator receive configured write permissions."""
        if agent.role_kind == "root":
            return None, "read-only"
        return self.config.claude_permission_mode, self.config.codex_sandbox

    async def _execute_turn(self, agent: AgentRecord, user_input: str) -> None:
        runner = self._runners.get(agent.provider)
        if runner is None:
            _, events = self.repo.fail_turn(agent.id, f"Unknown provider: {agent.provider}")
            self.event_bus.publish(events)
            await self.pump()
            return

        existing_children = [c for c in self.repo.list_children(agent.id) if c.role_kind == "worker"]
        remaining_questions = max(0, self.config.max_questions_per_turn - agent.question_count)
        prompt = build_agent_prompt(
            agent=agent,
            user_input=user_input,
            existing_children=existing_children,
            available_providers=self.available_providers(),
            max_depth=self.config.max_depth,
            remaining_questions=remaining_questions,
            integrator_available=self.worktree_mgr is not None,
            base_branch=self.git_ctx.base_branch,
        )

        async def _on_progress(line: str) -> None:
            cleaned = line.strip()
            if not cleaned:
                return
            _, prog_events = self.repo.append_progress(agent.id, cleaned)
            self.event_bus.publish(prog_events)

        turn_cwd = Path(agent.worktree_path) if agent.worktree_path else self.config.cwd
        claude_perm, codex_sandbox = self._turn_permissions(agent)

        try:
            output = await runner.run_turn(
                TurnInput(
                    agent=agent,
                    prompt=prompt,
                    cwd=turn_cwd,
                    timeout_seconds=self.config.turn_timeout_seconds,
                    claude_permission_mode=claude_perm,
                    codex_sandbox=codex_sandbox,
                    codex_network_access=self.config.codex_network_access,
                ),
                on_progress=_on_progress,
            )
            action = parse_agent_action(output.text)
            await self._apply_action(agent, action, output.provider_session_id)
        except RunnerError as exc:
            _, events = self.repo.fail_turn(agent.id, str(exc))
            self.event_bus.publish(events)
        except Exception as exc:  # pragma: no cover - defensive fallback
            _, events = self.repo.fail_turn(agent.id, f"Unexpected turn error: {exc}")
            self.event_bus.publish(events)
        finally:
            if not self._closed:
                await self.pump()

    def _post_worker_reply_git_sync(self, agent: AgentRecord) -> None:
        """Auto-commit worker worktree changes and enqueue onto Single-Writer Merge Queue."""
        if self.worktree_mgr is None or agent.role_kind != "worker":
            return
        if not agent.worktree_path or not agent.branch_name:
            return
        wt_path = Path(agent.worktree_path)
        try:
            committed = self.worktree_mgr.auto_commit_worktree(
                wt_path,
                f"peanuts({agent.theme}): agent update",
            )
            if committed:
                _, prog_ev = self.repo.append_progress(
                    agent.id,
                    f"Committed changes to branch {agent.branch_name}",
                )
                self.event_bus.publish(prog_ev)
            if self.config.auto_merge and self.worktree_mgr.has_commits_ahead(agent.branch_name):
                _, mq_evs = self.repo.enqueue_merge_job(
                    worker_agent_id=agent.id,
                    theme=agent.theme,
                    source_branch=agent.branch_name,
                    target_branch=self.worktree_mgr.base_branch,
                )
                self.event_bus.publish(mq_evs)
        except GitError as exc:
            _, prog_ev = self.repo.append_progress(agent.id, f"Git sync warning: {exc}")
            self.event_bus.publish(prog_ev)

    async def _apply_action(
        self,
        agent: AgentRecord,
        action: ReplyAction | DelegateAction | AskHumanAction | IntegrateAction,
        provider_session_id: str | None,
    ) -> None:
        if isinstance(action, ReplyAction):
            self._post_worker_reply_git_sync(agent)
            _, _, events = self.repo.finish_turn_reply(
                agent.id,
                action.message,
                provider_session_id=provider_session_id,
            )
            self.event_bus.publish(events)
            return

        if isinstance(action, IntegrateAction):
            note = self._handle_integrate_action(agent, action)
            _, _, events = self.repo.finish_turn_reply(
                agent.id,
                note,
                provider_session_id=provider_session_id,
            )
            self.event_bus.publish(events)
            return

        if isinstance(action, DelegateAction):
            if agent.depth >= self.config.max_depth:
                fallback = (
                    f"[Depth limit {self.config.max_depth} reached; cannot delegate further] "
                    f"{action.message or action.tasks[0].prompt}"
                )
                self._post_worker_reply_git_sync(agent)
                _, _, events = self.repo.finish_turn_reply(
                    agent.id,
                    fallback,
                    provider_session_id=provider_session_id,
                )
                self.event_bus.publish(events)
                return

            allowed = set(self.available_providers())
            default_child_provider = self.config.resolve_child_provider()
            if default_child_provider not in allowed:
                default_child_provider = self.available_providers()[0]
            task_specs: list[tuple[str, str, ProviderName, str | None]] = [
                (
                    t.theme,
                    t.prompt,
                    t.provider if (t.provider is not None and t.provider in allowed) else default_child_provider,
                    t.agent_id,
                )
                for t in action.tasks
            ]
            _, _, events = self.repo.delegate_batch(
                agent.id,
                task_specs,
                note=action.message,
                provider_session_id=provider_session_id,
                max_depth=self.config.max_depth,
            )
            self.event_bus.publish(events)
            return

        if isinstance(action, AskHumanAction):
            if agent.question_count >= self.config.max_questions_per_turn:
                fallback = (
                    f"[Question budget ({self.config.max_questions_per_turn}) exhausted; returning final reply] "
                    f"{action.question}"
                )
                self._post_worker_reply_git_sync(agent)
                _, _, events = self.repo.finish_turn_reply(
                    agent.id,
                    fallback,
                    provider_session_id=provider_session_id,
                )
                self.event_bus.publish(events)
                return

            _, _, events = self.repo.ask_human(
                agent.id,
                action.question,
                provider_session_id=provider_session_id,
                max_questions_per_turn=self.config.max_questions_per_turn,
            )
            self.event_bus.publish(events)
            return

    def _handle_integrate_action(self, agent: AgentRecord, action: IntegrateAction) -> str:
        """Validate and enqueue a root-dispatched IntegrateAction onto the Single-Writer queue."""
        if agent.role_kind != "root":
            return "[integrate rejected: only the root mediator may dispatch Integrator jobs]"
        if self.worktree_mgr is None:
            return "[integrate unavailable: git worktree / Integrator mode is not active in this workspace]"

        if action.mode == "resume":
            sup_evs = self.repo.supersede_integrator_dirty_questions()
            res_evs = self.repo.resume_waiting_merge_jobs()
            self.event_bus.publish(sup_evs + res_evs)
            return action.message or "Resumed paused Integrator merge queue."

        if action.mode == "merge":
            workers = [a for a in self.repo.list_agents() if a.role_kind == "worker"]
            target: AgentRecord | None = None
            if action.agent_id:
                target = next((w for w in workers if w.id == action.agent_id), None)
            if target is None and action.branch:
                target = next((w for w in workers if w.branch_name == action.branch), None)
            if target is None:
                return f"[integrate: target worker not found (agent_id={action.agent_id}, branch={action.branch})]"
            if target.status == AgentStatus.RUNNING:
                return f"Worker '{target.theme}' ({target.id}) is still RUNNING; merge will run once it finishes."
            if not target.branch_name or not target.worktree_path:
                return f"[integrate: worker '{target.theme}' has no isolated branch]"
            self.worktree_mgr.auto_commit_worktree(
                Path(target.worktree_path),
                f"peanuts({target.theme}): commit before root-triggered merge",
            )
            job, evs = self.repo.enqueue_merge_job(
                worker_agent_id=target.id,
                theme=target.theme,
                source_branch=target.branch_name,
                target_branch=self.worktree_mgr.base_branch,
            )
            self.event_bus.publish(evs)
            return action.message or f"Queued merge job {job.id} (`{job.source_branch}` -> `{job.target_branch}`)."

        # mode == "instruction"
        inst = action.instruction.strip()
        if not inst:
            return "[integrate: instruction mode requires a non-empty `instruction`]"
        job, evs = self.repo.enqueue_instruction_job(
            instruction=inst,
            target_branch=self.worktree_mgr.base_branch,
        )
        self.event_bus.publish(evs)
        return action.message or f"Queued Integrator instruction job {job.id} on `{job.target_branch}`."

    # ------------------------------------------------------------------
    # Single-Writer Integrator Execution (spec/PeanutsMergeQueue.tla)
    # ------------------------------------------------------------------

    async def _run_merge_job(self, job: MergeJobRecord) -> None:
        if self.worktree_mgr is None:
            return
        mgr = self.worktree_mgr
        integrator = self.repo.get_agent("integrator")
        if integrator is None:
            return

        if job.kind == "instruction":
            await self._run_instruction_job(integrator, job)
            return

        try:
            # 1. NeverMergeIntoDirtyBase: check parent's working tree before touching git
            if mgr.is_dirty(mgr.repo_root):
                _, pause_evs = self.repo.transition_merge_job(
                    job.id,
                    MergeJobStatus.WAITING_CLEAN_BASE,
                    summary="Paused: uncommitted changes in parent branch working tree",
                )
                self.event_bus.publish(pause_evs)
                _, prog_ev = self.repo.append_progress(
                    integrator.id,
                    f"Paused merge of {job.source_branch}: dirty working tree on {mgr.base_branch}",
                )
                self.event_bus.publish(prog_ev)

                # Ask human once if integrator isn't already waiting on an open question
                open_int_qs = [q for q in self.repo.list_questions(only_open=True) if q.agent_id == integrator.id]
                if not open_int_qs and integrator.status == AgentStatus.IDLE:
                    self.event_bus.publish(self.repo.set_integrator_status(AgentStatus.RUNNING))
                    _, _, q_evs = self.repo.ask_human(
                        integrator.id,
                        f"親ブランチ '{mgr.base_branch}' に未コミットの変更があるため、子ブランチ "
                        f"'{job.source_branch}' ({job.theme}) のマージを待機しています。"
                        "コミットまたは stash 後、`/answer ok` でマージを再開してください。",
                        max_questions_per_turn=self.config.max_questions_per_turn,
                    )
                    self.event_bus.publish(q_evs)
                return

            # 2. Acquire Single-Writer MERGING state
            self.event_bus.publish(self.repo.set_integrator_status(AgentStatus.RUNNING))
            _, start_evs = self.repo.transition_merge_job(job.id, MergeJobStatus.MERGING)
            self.event_bus.publish(start_evs)
            _, prog_ev = self.repo.append_progress(
                integrator.id,
                f"Merging {job.source_branch} -> {job.target_branch} ({job.theme})",
            )
            self.event_bus.publish(prog_ev)

            commit_msg = f"peanuts: merge {job.theme} ({job.source_branch})"
            attempt = await asyncio.to_thread(mgr.try_merge, job.source_branch, commit_msg)

            if attempt.outcome == "merged":
                _, done_evs = self.repo.transition_merge_job(
                    job.id,
                    MergeJobStatus.MERGED,
                    summary=attempt.summary,
                )
                self.event_bus.publish(done_evs)
                self.event_bus.publish(self.repo.set_integrator_status(AgentStatus.IDLE))
                notify_evs = self.repo.notify_root_integration(
                    f"Integrator merged branch `{job.source_branch}` (theme={job.theme}) into `{job.target_branch}` cleanly.",
                    theme=job.theme,
                )
                self.event_bus.publish(notify_evs)
                return

            if attempt.outcome == "conflict":
                _, conf_evs = self.repo.transition_merge_job(
                    job.id,
                    MergeJobStatus.CONFLICT_RESOLVING,
                    conflicted_files=attempt.conflicted_files,
                    summary=f"Resolving conflicts in: {', '.join(attempt.conflicted_files)}",
                )
                self.event_bus.publish(conf_evs)
                _, cprog_ev = self.repo.append_progress(
                    integrator.id,
                    f"Conflict in {', '.join(attempt.conflicted_files) or 'merge'}; invoking Integrator AI",
                )
                self.event_bus.publish(cprog_ev)

                resolved = await self._resolve_conflicts_with_integrator(
                    integrator,
                    job,
                    attempt.conflicted_files,
                    commit_msg,
                )
                if resolved:
                    _, done_evs = self.repo.transition_merge_job(
                        job.id,
                        MergeJobStatus.MERGED,
                        conflicted_files=[],
                        summary=f"Merged {job.source_branch} after AI conflict resolution",
                    )
                    self.event_bus.publish(done_evs)
                    self.event_bus.publish(self.repo.set_integrator_status(AgentStatus.IDLE))
                    notify_evs = self.repo.notify_root_integration(
                        f"Integrator resolved conflicts and merged `{job.source_branch}` (theme={job.theme}) into `{job.target_branch}`.",
                        theme=job.theme,
                    )
                    self.event_bus.publish(notify_evs)
                else:
                    await asyncio.to_thread(mgr.abort_merge)
                    _, abort_evs = self.repo.transition_merge_job(
                        job.id,
                        MergeJobStatus.ABORTED,
                        error="Could not auto-resolve conflicts; ran git merge --abort",
                    )
                    self.event_bus.publish(abort_evs)
                    self.event_bus.publish(self.repo.set_integrator_status(AgentStatus.IDLE))
                    notify_evs = self.repo.notify_root_integration(
                        f"Integrator aborted merge of `{job.source_branch}` (theme={job.theme}) due to unresolved conflicts in "
                        f"{', '.join(attempt.conflicted_files)}. Base branch `{job.target_branch}` was safely rolled back.",
                        theme=job.theme,
                    )
                    self.event_bus.publish(notify_evs)
                return

            # Merge error
            await asyncio.to_thread(mgr.abort_merge)
            _, err_evs = self.repo.transition_merge_job(
                job.id,
                MergeJobStatus.ABORTED,
                error=attempt.summary,
            )
            self.event_bus.publish(err_evs)
            self.event_bus.publish(self.repo.set_integrator_status(AgentStatus.IDLE))
            notify_evs = self.repo.notify_root_integration(
                f"Integrator failed to merge `{job.source_branch}`: {attempt.summary}",
                theme=job.theme,
            )
            self.event_bus.publish(notify_evs)
        except Exception as exc:
            await asyncio.to_thread(mgr.abort_merge)
            current = self.repo.get_merge_job(job.id)
            if current is not None and current.status not in {MergeJobStatus.MERGED, MergeJobStatus.ABORTED}:
                _, abort_evs = self.repo.transition_merge_job(
                    job.id,
                    MergeJobStatus.ABORTED,
                    error=str(exc),
                )
                self.event_bus.publish(abort_evs)
            self.event_bus.publish(self.repo.set_integrator_status(AgentStatus.IDLE))
        except asyncio.CancelledError:
            await asyncio.to_thread(mgr.abort_merge)
            raise

    async def _run_instruction_job(self, integrator: AgentRecord, job: MergeJobRecord) -> None:
        """Execute a root-dispatched git housekeeping instruction on the parent branch (Single-Writer)."""
        if self.worktree_mgr is None:
            return
        mgr = self.worktree_mgr
        runner = self._runners.get(integrator.provider)
        if runner is None:
            _, abort_evs = self.repo.transition_merge_job(
                job.id,
                MergeJobStatus.ABORTED,
                error=f"Unknown integrator provider: {integrator.provider}",
            )
            self.event_bus.publish(abort_evs)
            return

        try:
            # TLA+ IntegratorStartInstruction: atomically close stale dirty-base question before RUNNING
            self.event_bus.publish(self.repo.supersede_integrator_dirty_questions())
            self.event_bus.publish(self.repo.set_integrator_status(AgentStatus.RUNNING))
            _, start_evs = self.repo.transition_merge_job(job.id, MergeJobStatus.RUNNING_INSTRUCTION)
            self.event_bus.publish(start_evs)
            _, prog_ev = self.repo.append_progress(
                integrator.id,
                f"Running root instruction on {job.target_branch}: {(job.instruction or '')[:140]}",
            )
            self.event_bus.publish(prog_ev)

            prompt = (
                f"You are the dedicated Peanuts Integrator agent (the sole writer on branch `{job.target_branch}`).\n"
                f"Execute the following git/workspace instruction dispatched by the Root agent:\n\n"
                f"{job.instruction}\n\n"
                "Rules:\n"
                "- Do not leave an unfinished merge (no MERGE_HEAD) when done.\n"
                "- Report concisely what commands you ran and the resulting git status."
            )

            async def _on_progress(line: str) -> None:
                cleaned = line.strip()
                if cleaned:
                    _, evs = self.repo.append_progress(integrator.id, cleaned)
                    self.event_bus.publish(evs)

            claude_perm, codex_sandbox = self._turn_permissions(integrator)
            output = await runner.run_turn(
                TurnInput(
                    agent=integrator,
                    prompt=prompt,
                    cwd=mgr.repo_root,
                    timeout_seconds=self.config.turn_timeout_seconds,
                    claude_permission_mode=claude_perm,
                    codex_sandbox=codex_sandbox,
                    codex_network_access=self.config.codex_network_access,
                ),
                on_progress=_on_progress,
            )

            # Enforce CleanBaseWhenIntegratorIdle: never leave an unfinished MERGE_HEAD
            if mgr.is_merge_in_progress():
                await asyncio.to_thread(mgr.abort_merge)
                _, abort_evs = self.repo.transition_merge_job(
                    job.id,
                    MergeJobStatus.ABORTED,
                    error="Instruction left an unresolved merge; rolled back with git merge --abort",
                )
                self.event_bus.publish(abort_evs)
                self.event_bus.publish(self.repo.set_integrator_status(AgentStatus.IDLE))
                self.event_bus.publish(
                    self.repo.notify_root_integration(
                        f"Integrator instruction {job.id} aborted (left unresolved MERGE_HEAD; rolled back).",
                        theme=job.theme,
                    )
                )
                return

            _, done_evs = self.repo.transition_merge_job(
                job.id,
                MergeJobStatus.MERGED,
                summary=output.text[:500],
            )
            self.event_bus.publish(done_evs)
            self.event_bus.publish(self.repo.set_integrator_status(AgentStatus.IDLE))

            # If base branch is now clean, automatically resume any paused BRANCH_MERGE jobs!
            if not mgr.is_dirty(mgr.repo_root):
                self.event_bus.publish(self.repo.resume_waiting_merge_jobs())

            self.event_bus.publish(
                self.repo.notify_root_integration(
                    f"Integrator completed instruction on `{job.target_branch}`:\n{output.text}",
                    theme=job.theme,
                )
            )
        except asyncio.CancelledError:
            await asyncio.to_thread(mgr.abort_merge)
            raise
        except Exception as exc:
            await asyncio.to_thread(mgr.abort_merge)
            current = self.repo.get_merge_job(job.id)
            if current is not None and current.status not in {MergeJobStatus.MERGED, MergeJobStatus.ABORTED}:
                _, abort_evs = self.repo.transition_merge_job(
                    job.id,
                    MergeJobStatus.ABORTED,
                    error=str(exc),
                )
                self.event_bus.publish(abort_evs)
            self.event_bus.publish(self.repo.set_integrator_status(AgentStatus.IDLE))
            self.event_bus.publish(
                self.repo.notify_root_integration(
                    f"Integrator instruction {job.id} failed: {exc}",
                    theme=job.theme,
                )
            )

    async def _resolve_conflicts_with_integrator(
        self,
        integrator: AgentRecord,
        job: MergeJobRecord,
        conflicted_files: list[str],
        commit_msg: str,
    ) -> bool:
        if self.worktree_mgr is None:
            return False
        runner = self._runners.get(integrator.provider)
        if runner is None:
            return False

        files_list = "\n".join(f"- {f}" for f in conflicted_files) or "- (check `git status`)"
        prompt = (
            f"You are the dedicated Peanuts Integrator agent merging branch `{job.source_branch}` "
            f"(theme: {job.theme}) into `{job.target_branch}`.\n"
            f"`git merge --no-ff {job.source_branch}` produced conflicts in:\n{files_list}\n\n"
            "Instructions:\n"
            "1. Inspect the conflict markers in those files.\n"
            "2. Resolve all conflicts cleanly, preserving the intent of both branches.\n"
            "3. Stage the resolved files (`git add <files>`) and ensure no conflict markers remain.\n"
            "4. Briefly summarize how you resolved the conflicts."
        )

        async def _on_progress(line: str) -> None:
            cleaned = line.strip()
            if cleaned:
                _, evs = self.repo.append_progress(integrator.id, cleaned)
                self.event_bus.publish(evs)

        claude_perm, codex_sandbox = self._turn_permissions(integrator)
        try:
            await runner.run_turn(
                TurnInput(
                    agent=integrator,
                    prompt=prompt,
                    cwd=self.worktree_mgr.repo_root,
                    timeout_seconds=self.config.turn_timeout_seconds,
                    claude_permission_mode=claude_perm,
                    codex_sandbox=codex_sandbox,
                    codex_network_access=self.config.codex_network_access,
                ),
                on_progress=_on_progress,
            )
            return await asyncio.to_thread(self.worktree_mgr.finalize_resolved_merge, commit_msg)
        except (RunnerError, GitError):
            return False
