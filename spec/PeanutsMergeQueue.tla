------------------------- MODULE PeanutsMergeQueue -------------------------
EXTENDS Integers, FiniteSets, TLC

\* Formal model of Peanuts' Single-Writer Integrator + Root `integrate` action:
\*   - Child workers edit isolated git worktrees concurrently.
\*   - Two job kinds share ONE serialized queue drained by the singleton Integrator:
\*       1. "BRANCH_MERGE": merges a worker's branch into parent's branch (requires ~baseDirty)
\*       2. "INSTRUCTION": root-dispatched git housekeeping on parent's branch (e.g. git add/commit)
\*          which IS allowed when baseDirty = TRUE and can clean baseDirty to unblock paused merges!
\*   - Also models `integratorStatus` + `dirtyQuestionOpen` to catch deadlocks or
\*     WaitingHumanConsistency violations when root dispatches an INSTRUCTION while
\*     Integrator was paused waiting on a human dirty-base question.
CONSTANTS
    Workers,        \* e.g. {"w1", "w2"}
    MaxJobs         \* e.g. 2

JobIds == 1..MaxJobs
NONE == "NONE"

WorkerStates == {"IDLE", "WORKING"}
JobKinds == {"NONE", "BRANCH_MERGE", "INSTRUCTION"}
JobStatuses == {
    "NONE",
    "QUEUED",
    "WAITING_CLEAN_BASE",
    "MERGING",
    "CONFLICT_RESOLVING",
    "RUNNING_INSTRUCTION",
    "MERGED",
    "ABORTED"
}
IntegratorStatuses == {"IDLE", "RUNNING", "WAITING_HUMAN"}

ActiveJobStatuses == {"MERGING", "CONFLICT_RESOLVING", "RUNNING_INSTRUCTION"}
PendingOrActiveStatuses == {"QUEUED", "WAITING_CLEAN_BASE", "MERGING", "CONFLICT_RESOLVING", "RUNNING_INSTRUCTION"}
SettledStatuses == {"MERGED", "ABORTED"}

VARIABLES
    workerState,        \* [Workers -> WorkerStates]
    workerBranch,       \* [Workers -> Workers \cup {NONE}]
    jobKind,            \* [JobIds -> JobKinds]
    jobStatus,          \* [JobIds -> JobStatuses]
    jobWorker,          \* [JobIds -> Workers \cup {NONE}]
    nextJobId,          \* 1..(MaxJobs + 1)
    integratorStatus,   \* IntegratorStatuses
    dirtyQuestionOpen,  \* BOOLEAN (open ask_human question on Integrator for dirty base)
    baseDirty,          \* BOOLEAN (uncommitted edits on parent branch)
    baseMergeInProgress,\* BOOLEAN (MERGE_HEAD present on parent branch)
    humanDirtyCount     \* 0..1

vars == <<
    workerState, workerBranch, jobKind, jobStatus, jobWorker, nextJobId,
    integratorStatus, dirtyQuestionOpen, baseDirty, baseMergeInProgress, humanDirtyCount
>>

ActiveJobs == { j \in JobIds : jobStatus[j] \in ActiveJobStatuses }

\* ---------------------------------------------------------------------------
\* Initial State
\* ---------------------------------------------------------------------------
Init ==
    /\ workerState = [w \in Workers |-> "IDLE"]
    /\ workerBranch = [w \in Workers |-> NONE]
    /\ jobKind = [j \in JobIds |-> "NONE"]
    /\ jobStatus = [j \in JobIds |-> "NONE"]
    /\ jobWorker = [j \in JobIds |-> NONE]
    /\ nextJobId = 1
    /\ integratorStatus = "IDLE"
    /\ dirtyQuestionOpen = FALSE
    /\ baseDirty = FALSE
    /\ baseMergeInProgress = FALSE
    /\ humanDirtyCount = 0

\* ---------------------------------------------------------------------------
\* Actions
\* ---------------------------------------------------------------------------

\* 1. Human makes uncommitted edits on parent's working tree (when Integrator is not running)
HumanEditsBase ==
    /\ humanDirtyCount < 1
    /\ ~baseDirty
    /\ integratorStatus # "RUNNING"
    /\ ~baseMergeInProgress
    /\ baseDirty' = TRUE
    /\ humanDirtyCount' = humanDirtyCount + 1
    /\ UNCHANGED <<workerState, workerBranch, jobKind, jobStatus, jobWorker, nextJobId,
                   integratorStatus, dirtyQuestionOpen, baseMergeInProgress>>

\* 2. Human manually commits/stashes uncommitted edits & answers dirty-base question if open
HumanCleansBase ==
    /\ baseDirty
    /\ integratorStatus # "RUNNING"
    /\ baseDirty' = FALSE
    /\ dirtyQuestionOpen' = FALSE
    /\ integratorStatus' = IF integratorStatus = "WAITING_HUMAN" THEN "IDLE" ELSE integratorStatus
    /\ UNCHANGED <<workerState, workerBranch, jobKind, jobStatus, jobWorker, nextJobId,
                   baseMergeInProgress, humanDirtyCount>>

\* 3. Child worker starts in its own isolated git worktree
WorkerStart(w) ==
    /\ workerState[w] = "IDLE"
    /\ nextJobId + Cardinality({ x \in Workers : workerState[x] = "WORKING" }) <= MaxJobs
    /\ workerState' = [workerState EXCEPT ![w] = "WORKING"]
    /\ workerBranch' = [workerBranch EXCEPT ![w] = w]
    /\ UNCHANGED <<jobKind, jobStatus, jobWorker, nextJobId, integratorStatus,
                   dirtyQuestionOpen, baseDirty, baseMergeInProgress, humanDirtyCount>>

\* 4. Child worker commits to its branch and enqueues a BRANCH_MERGE job
WorkerCommitAndEnqueue(w) ==
    /\ workerState[w] = "WORKING"
    /\ nextJobId <= MaxJobs
    /\ LET j == nextJobId
       IN /\ workerState' = [workerState EXCEPT ![w] = "IDLE"]
          /\ jobKind' = [jobKind EXCEPT ![j] = "BRANCH_MERGE"]
          /\ jobStatus' = [jobStatus EXCEPT ![j] = "QUEUED"]
          /\ jobWorker' = [jobWorker EXCEPT ![j] = w]
          /\ nextJobId' = nextJobId + 1
    /\ UNCHANGED <<workerBranch, integratorStatus, dirtyQuestionOpen,
                   baseDirty, baseMergeInProgress, humanDirtyCount>>

\* 5. Root agent emits `integrate(mode="instruction")` at any time (even when base is clean)
RootEnqueueInstruction ==
    /\ nextJobId + Cardinality({ x \in Workers : workerState[x] = "WORKING" }) <= MaxJobs
    /\ LET j == nextJobId
       IN /\ jobKind' = [jobKind EXCEPT ![j] = "INSTRUCTION"]
          /\ jobStatus' = [jobStatus EXCEPT ![j] = "QUEUED"]
          /\ jobWorker' = [jobWorker EXCEPT ![j] = NONE]
          /\ nextJobId' = nextJobId + 1
    /\ UNCHANGED <<workerState, workerBranch, integratorStatus, dirtyQuestionOpen,
                   baseDirty, baseMergeInProgress, humanDirtyCount>>

\* 6a. Integrator sees dirty base on a BRANCH_MERGE -> pauses job & opens dirty-base question
IntegratorPauseForDirtyBase(j) ==
    /\ jobKind[j] = "BRANCH_MERGE"
    /\ jobStatus[j] = "QUEUED"
    /\ integratorStatus # "RUNNING"
    /\ baseDirty
    /\ jobStatus' = [jobStatus EXCEPT ![j] = "WAITING_CLEAN_BASE"]
    /\ dirtyQuestionOpen' = TRUE
    /\ integratorStatus' = "WAITING_HUMAN"
    /\ UNCHANGED <<workerState, workerBranch, jobKind, jobWorker, nextJobId,
                   baseDirty, baseMergeInProgress, humanDirtyCount>>

\* 6b. Base is clean again -> WAITING_CLEAN_BASE returns to QUEUED
ResumeQueuedAfterCleanBase(j) ==
    /\ jobKind[j] = "BRANCH_MERGE"
    /\ jobStatus[j] = "WAITING_CLEAN_BASE"
    /\ ~baseDirty
    /\ jobStatus' = [jobStatus EXCEPT ![j] = "QUEUED"]
    /\ UNCHANGED <<workerState, workerBranch, jobKind, jobWorker, nextJobId,
                   integratorStatus, dirtyQuestionOpen, baseDirty, baseMergeInProgress, humanDirtyCount>>

\* 7. Integrator acquires Single-Writer lock for a BRANCH_MERGE (requires ~baseDirty & IDLE)
IntegratorStartMerge(j) ==
    /\ jobKind[j] = "BRANCH_MERGE"
    /\ jobStatus[j] = "QUEUED"
    /\ integratorStatus = "IDLE"
    /\ ~dirtyQuestionOpen
    /\ ~baseDirty
    /\ ~baseMergeInProgress
    /\ integratorStatus' = "RUNNING"
    /\ baseMergeInProgress' = TRUE
    /\ jobStatus' = [jobStatus EXCEPT ![j] = "MERGING"]
    /\ UNCHANGED <<workerState, workerBranch, jobKind, jobWorker, nextJobId,
                   dirtyQuestionOpen, baseDirty, humanDirtyCount>>

\* 8. Integrator acquires Single-Writer lock for a root INSTRUCTION (allowed even if baseDirty
\*    or WAITING_HUMAN on dirty-base question; supersedes stale dirty-base question atomically!)
IntegratorStartInstruction(j) ==
    /\ jobKind[j] = "INSTRUCTION"
    /\ jobStatus[j] = "QUEUED"
    /\ integratorStatus \in {"IDLE", "WAITING_HUMAN"}
    /\ ~baseMergeInProgress
    /\ integratorStatus' = "RUNNING"
    /\ dirtyQuestionOpen' = FALSE
    /\ jobStatus' = [jobStatus EXCEPT ![j] = "RUNNING_INSTRUCTION"]
    /\ UNCHANGED <<workerState, workerBranch, jobKind, jobWorker, nextJobId,
                   baseDirty, baseMergeInProgress, humanDirtyCount>>

\* 9a. Instruction succeeds and cleans baseDirty (e.g. git add && git commit)
InstructionSuccessClean(j) ==
    /\ jobKind[j] = "INSTRUCTION"
    /\ jobStatus[j] = "RUNNING_INSTRUCTION"
    /\ integratorStatus = "RUNNING"
    /\ ~baseMergeInProgress
    /\ jobStatus' = [jobStatus EXCEPT ![j] = "MERGED"]
    /\ integratorStatus' = "IDLE"
    /\ baseDirty' = FALSE
    /\ UNCHANGED <<workerState, workerBranch, jobKind, jobWorker, nextJobId,
                   dirtyQuestionOpen, baseMergeInProgress, humanDirtyCount>>

\* 9b. Instruction fails or leaves uncommitted dirt on base (never leaves MERGE_HEAD)
InstructionFailedOrStillDirty(j) ==
    /\ jobKind[j] = "INSTRUCTION"
    /\ jobStatus[j] = "RUNNING_INSTRUCTION"
    /\ integratorStatus = "RUNNING"
    /\ ~baseMergeInProgress
    /\ jobStatus' = [jobStatus EXCEPT ![j] = "ABORTED"]
    /\ integratorStatus' = "IDLE"
    /\ baseDirty' = TRUE
    /\ UNCHANGED <<workerState, workerBranch, jobKind, jobWorker, nextJobId,
                   dirtyQuestionOpen, baseMergeInProgress, humanDirtyCount>>

\* 10a. Clean branch merge succeeds
MergeCleanSuccess(j) ==
    /\ jobKind[j] = "BRANCH_MERGE"
    /\ jobStatus[j] = "MERGING"
    /\ integratorStatus = "RUNNING"
    /\ baseMergeInProgress
    /\ jobStatus' = [jobStatus EXCEPT ![j] = "MERGED"]
    /\ integratorStatus' = "IDLE"
    /\ baseMergeInProgress' = FALSE
    /\ UNCHANGED <<workerState, workerBranch, jobKind, jobWorker, nextJobId,
                   dirtyQuestionOpen, baseDirty, humanDirtyCount>>

\* 10b. Branch merge hits conflicts -> AI enters CONFLICT_RESOLVING
MergeHitsConflict(j) ==
    /\ jobKind[j] = "BRANCH_MERGE"
    /\ jobStatus[j] = "MERGING"
    /\ integratorStatus = "RUNNING"
    /\ baseMergeInProgress
    /\ jobStatus' = [jobStatus EXCEPT ![j] = "CONFLICT_RESOLVING"]
    /\ UNCHANGED <<workerState, workerBranch, jobKind, jobWorker, nextJobId,
                   integratorStatus, dirtyQuestionOpen, baseDirty, baseMergeInProgress, humanDirtyCount>>

\* 11a. Integrator AI resolves conflicts and commits
ConflictResolvedAndCommitted(j) ==
    /\ jobKind[j] = "BRANCH_MERGE"
    /\ jobStatus[j] = "CONFLICT_RESOLVING"
    /\ integratorStatus = "RUNNING"
    /\ baseMergeInProgress
    /\ jobStatus' = [jobStatus EXCEPT ![j] = "MERGED"]
    /\ integratorStatus' = "IDLE"
    /\ baseMergeInProgress' = FALSE
    /\ UNCHANGED <<workerState, workerBranch, jobKind, jobWorker, nextJobId,
                   dirtyQuestionOpen, baseDirty, humanDirtyCount>>

\* 11b. Integrator cannot resolve conflicts -> runs `git merge --abort`
ConflictAborted(j) ==
    /\ jobKind[j] = "BRANCH_MERGE"
    /\ jobStatus[j] = "CONFLICT_RESOLVING"
    /\ integratorStatus = "RUNNING"
    /\ baseMergeInProgress
    /\ jobStatus' = [jobStatus EXCEPT ![j] = "ABORTED"]
    /\ integratorStatus' = "IDLE"
    /\ baseMergeInProgress' = FALSE
    /\ UNCHANGED <<workerState, workerBranch, jobKind, jobWorker, nextJobId,
                   dirtyQuestionOpen, baseDirty, humanDirtyCount>>

\* 12. Valid quiescent terminal stuttering
QuiescentDone ==
    /\ nextJobId > MaxJobs
    /\ \A w \in Workers : workerState[w] = "IDLE"
    /\ \A j \in JobIds : jobStatus[j] \in SettledStatuses
    /\ ~baseDirty
    /\ integratorStatus = "IDLE"
    /\ ~dirtyQuestionOpen
    /\ ~baseMergeInProgress
    /\ UNCHANGED vars

\* ---------------------------------------------------------------------------
\* Next-State Relation & Fairness
\* ---------------------------------------------------------------------------
Next ==
    \/ HumanEditsBase
    \/ HumanCleansBase
    \/ \E w \in Workers : WorkerStart(w)
    \/ \E w \in Workers : WorkerCommitAndEnqueue(w)
    \/ RootEnqueueInstruction
    \/ \E j \in JobIds : IntegratorPauseForDirtyBase(j)
    \/ \E j \in JobIds : ResumeQueuedAfterCleanBase(j)
    \/ \E j \in JobIds : IntegratorStartMerge(j)
    \/ \E j \in JobIds : IntegratorStartInstruction(j)
    \/ \E j \in JobIds : InstructionSuccessClean(j)
    \/ \E j \in JobIds : InstructionFailedOrStillDirty(j)
    \/ \E j \in JobIds : MergeCleanSuccess(j)
    \/ \E j \in JobIds : MergeHitsConflict(j)
    \/ \E j \in JobIds : ConflictResolvedAndCommitted(j)
    \/ \E j \in JobIds : ConflictAborted(j)
    \/ QuiescentDone

Fairness ==
    /\ WF_vars(HumanCleansBase)
    /\ \A w \in Workers : WF_vars(WorkerCommitAndEnqueue(w))
    /\ \A j \in JobIds :
        /\ WF_vars(ResumeQueuedAfterCleanBase(j))
        /\ WF_vars(IntegratorStartMerge(j))
        /\ WF_vars(IntegratorStartInstruction(j))
        /\ WF_vars(InstructionSuccessClean(j))
        /\ WF_vars(MergeCleanSuccess(j))
        /\ WF_vars(ConflictResolvedAndCommitted(j))

Spec == Init /\ [][Next]_vars /\ Fairness

\* ---------------------------------------------------------------------------
\* Safety Invariants
\* ---------------------------------------------------------------------------
TypeOK ==
    /\ workerState \in [Workers -> WorkerStates]
    /\ workerBranch \in [Workers -> Workers \cup {NONE}]
    /\ jobKind \in [JobIds -> JobKinds]
    /\ jobStatus \in [JobIds -> JobStatuses]
    /\ jobWorker \in [JobIds -> Workers \cup {NONE}]
    /\ nextJobId \in 1..(MaxJobs + 1)
    /\ integratorStatus \in IntegratorStatuses
    /\ dirtyQuestionOpen \in BOOLEAN
    /\ baseDirty \in BOOLEAN
    /\ baseMergeInProgress \in BOOLEAN
    /\ humanDirtyCount \in 0..1

SingleIntegratorWriter ==
    /\ Cardinality(ActiveJobs) <= 1
    /\ (integratorStatus = "RUNNING") <=> (Cardinality(ActiveJobs) = 1)

IntegratorQuestionConsistency ==
    (integratorStatus = "WAITING_HUMAN") <=> dirtyQuestionOpen

NeverMergeIntoDirtyBase ==
    \A j \in JobIds :
        jobStatus[j] \in {"MERGING", "CONFLICT_RESOLVING"} => ~baseDirty

CleanBaseWhenIntegratorIdle ==
    (integratorStatus # "RUNNING") => (~baseMergeInProgress)

DisjointWorkerBranches ==
    \A w1, w2 \in Workers :
        (w1 # w2 /\ workerBranch[w1] # NONE /\ workerBranch[w2] # NONE)
        => workerBranch[w1] # workerBranch[w2]

\* ---------------------------------------------------------------------------
\* Liveness Properties
\* ---------------------------------------------------------------------------
AllMergeJobsEventuallySettle ==
    \A j \in JobIds :
        (jobStatus[j] \in PendingOrActiveStatuses) ~> (jobStatus[j] \in SettledStatuses)

BaseEventuallyClean ==
    baseMergeInProgress ~> (~baseMergeInProgress)

=============================================================================
