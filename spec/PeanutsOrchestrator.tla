----------------------- MODULE PeanutsOrchestrator -----------------------
EXTENDS Integers, Sequences, FiniteSets, TLC

\* Non-blocking Root Mediator + Bounded Recursive Child Delegation:
\*   - RootAgent (depth=0) runs an async event loop: when it delegates to a child,
\*     RootAgent immediately returns to IDLE so it can accept new human themes and
\*     spawn additional child agents while earlier children are still RUNNING.
\*   - Whenever a root-owned child finishes/fails/cancels, its result is enqueued
\*     into RootAgent's inbox so RootAgent wakes up and reports back to the human.
\*   - ChildAgents (depth>=1) wait in WAITING_CHILD when delegating to grandchildren
\*     so they only reply to their parent once their own subtasks have settled.
CONSTANTS
    RootAgent,
    ChildAgents,
    MaxDepth,
    MaxDelegations,
    MaxHumanRequests,
    MaxQuestionsPerTurn

Agents == {RootAgent} \cup ChildAgents
DelegationIds == 1..MaxDelegations
NONE == "NONE"

AgentStatuses == {"IDLE", "RUNNING", "WAITING_CHILD", "WAITING_HUMAN", "FAILED"}
DelegationStatuses == {"NONE", "PENDING", "IN_PROGRESS", "NEEDS_HUMAN", "DONE", "FAILED", "CANCELLED"}

VARIABLES
    agentAllocated,     \* [Agents -> BOOLEAN]
    agentStatus,        \* [Agents -> AgentStatuses]
    parentOf,           \* [Agents -> Agents \cup {NONE}]
    depth,              \* [Agents -> 0..MaxDepth]
    activeTurns,        \* [Agents -> 0..1] (single CLI --resume lock per agent)
    inboxCount,         \* [Agents -> Nat] (pending human msgs / answers / child results)
    pendingQuestion,    \* [Agents -> BOOLEAN]
    questionCount,      \* [Agents -> 0..MaxQuestionsPerTurn]
    delStatus,          \* [DelegationIds -> DelegationStatuses]
    delCaller,          \* [DelegationIds -> Agents \cup {NONE}]
    delCallee,          \* [DelegationIds -> Agents \cup {NONE}]
    nextDelId,          \* 1..(MaxDelegations + 1)
    humanReqCount       \* 0..MaxHumanRequests

vars == <<
    agentAllocated, agentStatus, parentOf, depth, activeTurns,
    inboxCount, pendingQuestion, questionCount, delStatus, delCaller, delCallee,
    nextDelId, humanReqCount
>>

ActiveDelStatuses == {"PENDING", "IN_PROGRESS", "NEEDS_HUMAN"}
SettledDelStatuses == {"DONE", "FAILED", "CANCELLED"}

ActiveOutgoingDels(a) ==
    { d \in DelegationIds : delCaller[d] = a /\ delStatus[d] \in ActiveDelStatuses }

ActiveIncomingDels(a) ==
    { d \in DelegationIds : delCallee[d] = a /\ delStatus[d] \in ActiveDelStatuses }

CanAssignChild(caller, c) ==
    /\ c \in ChildAgents
    /\ c # caller
    /\ ( (~agentAllocated[c])
         \/ (parentOf[c] = caller /\ depth[c] = depth[caller] + 1
             /\ ActiveIncomingDels(c) = {}
             /\ agentStatus[c] = "IDLE"
             /\ inboxCount[c] = 0) )

\* ---------------------------------------------------------------------------
\* Initial State
\* ---------------------------------------------------------------------------
Init ==
    /\ agentAllocated = [a \in Agents |-> (a = RootAgent)]
    /\ agentStatus = [a \in Agents |-> "IDLE"]
    /\ parentOf = [a \in Agents |-> NONE]
    /\ depth = [a \in Agents |-> 0]
    /\ activeTurns = [a \in Agents |-> 0]
    /\ inboxCount = [a \in Agents |-> 0]
    /\ pendingQuestion = [a \in Agents |-> FALSE]
    /\ questionCount = [a \in Agents |-> 0]
    /\ delStatus = [d \in DelegationIds |-> "NONE"]
    /\ delCaller = [d \in DelegationIds |-> NONE]
    /\ delCallee = [d \in DelegationIds |-> NONE]
    /\ nextDelId = 1
    /\ humanReqCount = 0

\* ---------------------------------------------------------------------------
\* Actions
\* ---------------------------------------------------------------------------

\* 1. Human sends a new message/theme to the Root Parent Agent at any time
HumanSendMessage ==
    /\ humanReqCount < MaxHumanRequests
    /\ humanReqCount' = humanReqCount + 1
    /\ inboxCount' = [inboxCount EXCEPT ![RootAgent] = @ + 1]
    /\ UNCHANGED <<agentAllocated, agentStatus, parentOf, depth, activeTurns,
                   pendingQuestion, questionCount, delStatus, delCaller, delCallee, nextDelId>>

\* 2. An IDLE agent with inbox work starts a turn (acquires single-turn CLI lock)
StartTurn(a) ==
    /\ agentAllocated[a]
    /\ agentStatus[a] = "IDLE"
    /\ inboxCount[a] > 0
    /\ activeTurns[a] = 0
    /\ agentStatus' = [agentStatus EXCEPT ![a] = "RUNNING"]
    /\ activeTurns' = [activeTurns EXCEPT ![a] = 1]
    /\ inboxCount' = [inboxCount EXCEPT ![a] = @ - 1]
    /\ UNCHANGED <<agentAllocated, parentOf, depth, pendingQuestion, questionCount,
                   delStatus, delCaller, delCallee, nextDelId, humanReqCount>>

\* 3. A RUNNING agent finishes its turn with Action = reply
\*    If it settles a delegation owned by RootAgent, enqueue a child-result event for RootAgent.
FinishTurnReply(a) ==
    /\ agentAllocated[a]
    /\ agentStatus[a] = "RUNNING"
    /\ activeTurns[a] = 1
    /\ agentStatus' = [agentStatus EXCEPT ![a] = "IDLE"]
    /\ activeTurns' = [activeTurns EXCEPT ![a] = 0]
    /\ questionCount' = [questionCount EXCEPT ![a] = 0]
    /\ LET inDels == { d \in DelegationIds : delCallee[d] = a /\ delStatus[d] = "IN_PROGRESS" }
           notifiesRoot == \E d \in inDels : delCaller[d] = RootAgent
       IN /\ delStatus' = [d \in DelegationIds |->
                            IF d \in inDels THEN "DONE" ELSE delStatus[d]]
          /\ inboxCount' = [x \in Agents |->
                             IF x = RootAgent /\ notifiesRoot
                             THEN inboxCount[x] + 1
                             ELSE inboxCount[x]]
    /\ UNCHANGED <<agentAllocated, parentOf, depth, pendingQuestion,
                   delCaller, delCallee, nextDelId, humanReqCount>>

\* 4a. A RUNNING agent delegates a subtask to 1 child agent:
\*     RootAgent returns immediately to IDLE (non-blocking event loop);
\*     non-root children transition to WAITING_CHILD until their grandchildren settle.
DelegateOne(a, c) ==
    /\ agentAllocated[a]
    /\ agentStatus[a] = "RUNNING"
    /\ activeTurns[a] = 1
    /\ depth[a] < MaxDepth
    /\ nextDelId <= MaxDelegations
    /\ CanAssignChild(a, c)
    /\ LET d == nextDelId
           nextCallerStatus == IF a = RootAgent THEN "IDLE" ELSE "WAITING_CHILD"
       IN /\ agentAllocated' = [agentAllocated EXCEPT ![c] = TRUE]
          /\ parentOf' = [parentOf EXCEPT ![c] = a]
          /\ depth' = [depth EXCEPT ![c] = depth[a] + 1]
          /\ agentStatus' = [agentStatus EXCEPT ![a] = nextCallerStatus]
          /\ activeTurns' = [activeTurns EXCEPT ![a] = 0]
          /\ delStatus' = [delStatus EXCEPT ![d] = "PENDING"]
          /\ delCaller' = [delCaller EXCEPT ![d] = a]
          /\ delCallee' = [delCallee EXCEPT ![d] = c]
          /\ nextDelId' = nextDelId + 1
    /\ UNCHANGED <<inboxCount, pendingQuestion, questionCount, humanReqCount>>

\* 4b. A RUNNING agent fans out in parallel to 2 distinct child agents in one turn
DelegateTwo(a, c1, c2) ==
    /\ agentAllocated[a]
    /\ agentStatus[a] = "RUNNING"
    /\ activeTurns[a] = 1
    /\ depth[a] < MaxDepth
    /\ nextDelId + 1 <= MaxDelegations
    /\ c1 # c2
    /\ CanAssignChild(a, c1)
    /\ CanAssignChild(a, c2)
    /\ LET d1 == nextDelId
           d2 == nextDelId + 1
           nextCallerStatus == IF a = RootAgent THEN "IDLE" ELSE "WAITING_CHILD"
       IN /\ agentAllocated' = [agentAllocated EXCEPT ![c1] = TRUE, ![c2] = TRUE]
          /\ parentOf' = [parentOf EXCEPT ![c1] = a, ![c2] = a]
          /\ depth' = [depth EXCEPT ![c1] = depth[a] + 1, ![c2] = depth[a] + 1]
          /\ agentStatus' = [agentStatus EXCEPT ![a] = nextCallerStatus]
          /\ activeTurns' = [activeTurns EXCEPT ![a] = 0]
          /\ delStatus' = [delStatus EXCEPT ![d1] = "PENDING", ![d2] = "PENDING"]
          /\ delCaller' = [delCaller EXCEPT ![d1] = a, ![d2] = a]
          /\ delCallee' = [delCallee EXCEPT ![d1] = c1, ![d2] = c2]
          /\ nextDelId' = nextDelId + 2
    /\ UNCHANGED <<inboxCount, pendingQuestion, questionCount, humanReqCount>>

\* 5. A child agent picks up a PENDING delegation and starts its turn
StartDelegation(d) ==
    /\ d \in DelegationIds
    /\ delStatus[d] = "PENDING"
    /\ LET c == delCallee[d]
       IN /\ agentAllocated[c]
          /\ agentStatus[c] = "IDLE"
          /\ activeTurns[c] = 0
          /\ delStatus' = [delStatus EXCEPT ![d] = "IN_PROGRESS"]
          /\ agentStatus' = [agentStatus EXCEPT ![c] = "RUNNING"]
          /\ activeTurns' = [activeTurns EXCEPT ![c] = 1]
          /\ questionCount' = [questionCount EXCEPT ![c] = 0]
    /\ UNCHANGED <<agentAllocated, parentOf, depth, inboxCount, pendingQuestion,
                   delCaller, delCallee, nextDelId, humanReqCount>>

\* 6. A RUNNING agent asks the human a clarification question (bounded by MaxQuestionsPerTurn)
AskHuman(a) ==
    /\ agentAllocated[a]
    /\ agentStatus[a] = "RUNNING"
    /\ activeTurns[a] = 1
    /\ ~pendingQuestion[a]
    /\ questionCount[a] < MaxQuestionsPerTurn
    /\ agentStatus' = [agentStatus EXCEPT ![a] = "WAITING_HUMAN"]
    /\ activeTurns' = [activeTurns EXCEPT ![a] = 0]
    /\ pendingQuestion' = [pendingQuestion EXCEPT ![a] = TRUE]
    /\ questionCount' = [questionCount EXCEPT ![a] = @ + 1]
    /\ delStatus' = [d \in DelegationIds |->
                      IF delCallee[d] = a /\ delStatus[d] = "IN_PROGRESS"
                      THEN "NEEDS_HUMAN"
                      ELSE delStatus[d]]
    /\ UNCHANGED <<agentAllocated, parentOf, depth, inboxCount,
                   delCaller, delCallee, nextDelId, humanReqCount>>

\* 7. Human answers a pending question -> transitions agent back to IDLE with answer in inbox
HumanAnswer(a) ==
    /\ agentAllocated[a]
    /\ agentStatus[a] = "WAITING_HUMAN"
    /\ pendingQuestion[a]
    /\ pendingQuestion' = [pendingQuestion EXCEPT ![a] = FALSE]
    /\ agentStatus' = [agentStatus EXCEPT ![a] = "IDLE"]
    /\ inboxCount' = [inboxCount EXCEPT ![a] = @ + 1]
    /\ delStatus' = [d \in DelegationIds |->
                      IF delCallee[d] = a /\ delStatus[d] = "NEEDS_HUMAN"
                      THEN "IN_PROGRESS"
                      ELSE delStatus[d]]
    /\ UNCHANGED <<agentAllocated, parentOf, depth, activeTurns, questionCount,
                   delCaller, delCallee, nextDelId, humanReqCount>>

\* 8. A RUNNING agent fails -> transitions to FAILED, marks incoming delegation FAILED,
\*    and notifies RootAgent if RootAgent was the caller.
FailTurn(a) ==
    /\ agentAllocated[a]
    /\ agentStatus[a] = "RUNNING"
    /\ activeTurns[a] = 1
    /\ agentStatus' = [agentStatus EXCEPT ![a] = "FAILED"]
    /\ activeTurns' = [activeTurns EXCEPT ![a] = 0]
    /\ questionCount' = [questionCount EXCEPT ![a] = 0]
    /\ LET inDels == { d \in DelegationIds : delCallee[d] = a /\ delStatus[d] = "IN_PROGRESS" }
           notifiesRoot == \E d \in inDels : delCaller[d] = RootAgent
       IN /\ delStatus' = [d \in DelegationIds |->
                            IF d \in inDels THEN "FAILED" ELSE delStatus[d]]
          /\ inboxCount' = [x \in Agents |->
                             IF x = RootAgent /\ notifiesRoot
                             THEN inboxCount[x] + 1
                             ELSE inboxCount[x]]
    /\ UNCHANGED <<agentAllocated, parentOf, depth, pendingQuestion,
                   delCaller, delCallee, nextDelId, humanReqCount>>

\* 9. A FAILED agent is reset/retried back to IDLE
ResetFailed(a) ==
    /\ agentAllocated[a]
    /\ agentStatus[a] = "FAILED"
    /\ activeTurns[a] = 0
    /\ agentStatus' = [agentStatus EXCEPT ![a] = "IDLE"]
    /\ UNCHANGED <<agentAllocated, parentOf, depth, activeTurns, inboxCount,
                   pendingQuestion, questionCount, delStatus, delCaller, delCallee, nextDelId, humanReqCount>>

\* 10. Non-root parent resumes once ALL of its outgoing grandchild delegations settle
ResumeAfterChildren(a) ==
    /\ a # RootAgent
    /\ agentAllocated[a]
    /\ agentStatus[a] = "WAITING_CHILD"
    /\ ActiveOutgoingDels(a) = {}
    /\ activeTurns[a] = 0
    /\ agentStatus' = [agentStatus EXCEPT ![a] = "IDLE"]
    /\ inboxCount' = [inboxCount EXCEPT ![a] = @ + 1]
    /\ UNCHANGED <<agentAllocated, parentOf, depth, activeTurns, pendingQuestion, questionCount,
                   delStatus, delCaller, delCallee, nextDelId, humanReqCount>>

\* 11. Cancel a PENDING delegation before it starts
CancelPendingDelegation(d) ==
    /\ d \in DelegationIds
    /\ delStatus[d] = "PENDING"
    /\ delStatus' = [delStatus EXCEPT ![d] = "CANCELLED"]
    /\ inboxCount' = [x \in Agents |->
                       IF x = RootAgent /\ delCaller[d] = RootAgent
                       THEN inboxCount[x] + 1
                       ELSE inboxCount[x]]
    /\ UNCHANGED <<agentAllocated, agentStatus, parentOf, depth, activeTurns,
                   pendingQuestion, questionCount, delCaller, delCallee, nextDelId, humanReqCount>>

\* 12. Valid quiescent terminal stuttering once all bounded human inputs have been processed
AllQuiescent ==
    /\ \A a \in Agents : agentStatus[a] = "IDLE" /\ inboxCount[a] = 0 /\ ~pendingQuestion[a]
    /\ \A d \in DelegationIds : delStatus[d] \notin ActiveDelStatuses

QuiescentDone ==
    /\ humanReqCount = MaxHumanRequests
    /\ AllQuiescent
    /\ UNCHANGED vars

\* ---------------------------------------------------------------------------
\* Next-State Relation & Fairness
\* ---------------------------------------------------------------------------
Next ==
    \/ HumanSendMessage
    \/ \E a \in Agents : StartTurn(a)
    \/ \E a \in Agents : FinishTurnReply(a)
    \/ \E a \in Agents, c \in ChildAgents : DelegateOne(a, c)
    \/ \E a \in Agents, c1 \in ChildAgents, c2 \in ChildAgents : DelegateTwo(a, c1, c2)
    \/ \E d \in DelegationIds : StartDelegation(d)
    \/ \E a \in Agents : AskHuman(a)
    \/ \E a \in Agents : HumanAnswer(a)
    \/ \E a \in Agents : FailTurn(a)
    \/ \E a \in Agents : ResetFailed(a)
    \/ \E a \in Agents : ResumeAfterChildren(a)
    \/ \E d \in DelegationIds : CancelPendingDelegation(d)
    \/ QuiescentDone

Fairness ==
    /\ \A a \in Agents :
        /\ WF_vars(StartTurn(a))
        /\ WF_vars(FinishTurnReply(a))
        /\ WF_vars(HumanAnswer(a))
        /\ WF_vars(ResetFailed(a))
        /\ WF_vars(ResumeAfterChildren(a))
    /\ \A d \in DelegationIds :
        WF_vars(StartDelegation(d))

Spec == Init /\ [][Next]_vars /\ Fairness

\* ---------------------------------------------------------------------------
\* Safety Invariants
\* ---------------------------------------------------------------------------
TypeOK ==
    /\ agentAllocated \in [Agents -> BOOLEAN]
    /\ agentStatus \in [Agents -> AgentStatuses]
    /\ parentOf \in [Agents -> Agents \cup {NONE}]
    /\ depth \in [Agents -> 0..MaxDepth]
    /\ activeTurns \in [Agents -> 0..1]
    /\ \A a \in Agents : inboxCount[a] >= 0
    /\ pendingQuestion \in [Agents -> BOOLEAN]
    /\ questionCount \in [Agents -> 0..MaxQuestionsPerTurn]
    /\ delStatus \in [DelegationIds -> DelegationStatuses]
    /\ delCaller \in [DelegationIds -> Agents \cup {NONE}]
    /\ delCallee \in [DelegationIds -> Agents \cup {NONE}]
    /\ nextDelId \in 1..(MaxDelegations + 1)
    /\ humanReqCount \in 0..MaxHumanRequests

DepthBound ==
    \A a \in Agents : agentAllocated[a] => (depth[a] >= 0 /\ depth[a] <= MaxDepth)

StrictTreeHierarchy ==
    /\ parentOf[RootAgent] = NONE
    /\ depth[RootAgent] = 0
    /\ \A a \in ChildAgents :
        agentAllocated[a] =>
            (parentOf[a] \in Agents /\ parentOf[a] # a /\ depth[a] = depth[parentOf[a]] + 1)
    /\ \A d \in DelegationIds :
        delStatus[d] # "NONE" =>
            (delCaller[d] \in Agents /\ delCallee[d] \in ChildAgents /\ depth[delCaller[d]] < depth[delCallee[d]])

NoConcurrentResume ==
    \A a \in Agents :
        /\ (agentStatus[a] = "RUNNING") <=> (activeTurns[a] = 1)
        /\ activeTurns[a] <= 1

WaitingHumanConsistency ==
    \A a \in Agents :
        (agentStatus[a] = "WAITING_HUMAN") <=> pendingQuestion[a]

RootNeverBlockedOnChildren ==
    agentStatus[RootAgent] # "WAITING_CHILD"

ChildCallerWaitsForGrandchildren ==
    \A d \in DelegationIds :
        (delStatus[d] \in ActiveDelStatuses /\ delCaller[d] # RootAgent)
        => agentStatus[delCaller[d]] = "WAITING_CHILD"

AtMostOneActiveIncomingDelegation ==
    \A a \in Agents :
        Cardinality(ActiveIncomingDels(a)) <= 1

\* ---------------------------------------------------------------------------
\* Liveness Properties
\* ---------------------------------------------------------------------------
AllDelegationsEventuallySettle ==
    \A d \in DelegationIds :
        (delStatus[d] \in ActiveDelStatuses) ~> (delStatus[d] \in SettledDelStatuses)

AllQuestionsEventuallyResolved ==
    \A a \in Agents :
        pendingQuestion[a] ~> (~pendingQuestion[a])

WaitingParentEventuallyWakes ==
    \A a \in Agents :
        (agentStatus[a] = "WAITING_CHILD") ~> (agentStatus[a] # "WAITING_CHILD")

=============================================================================
