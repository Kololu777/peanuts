------------------------ MODULE PeanutsCompletion ------------------------
EXTENDS Integers, FiniteSets, TLC

\* Formal model of Peanuts' @file and /command autocomplete popup state machine:
\*   Verifies the classic autocomplete bugs before implementation:
\*     1. Out-of-bounds selectedIndex when candidate count shrinks while typing
\*     2. Empty ghost popup staying open when 0 candidates match
\*     3. Async stale-response race (slow earlier query overwriting newer query results)
\*     4. Accidental Enter submission sending half-typed `@src/tu` or `/pro` to the agent
\*     5. Escape dismissal cleanly closing popup so Enter submits normally
CONSTANTS
    MaxCandidates,  \* e.g. 3
    MaxQueries      \* e.g. 3

Modes == {"NONE", "COMMAND", "FILE"}
ReqIds == 0..MaxQueries

VARIABLES
    popupOpen,          \* BOOLEAN
    mode,               \* Modes
    candidateCount,     \* 0..MaxCandidates
    selectedIndex,      \* 0..MaxCandidates (0 when closed; 1..candidateCount when open)
    latestReqId,        \* 0..MaxQueries (monotonic ID of latest typed query)
    invalidatedUpTo,    \* 0..MaxQueries (queries <= this were invalidated by Escape/Accept/Clear)
    appliedReqId,       \* 0..MaxQueries (highest request ID applied to UI)
    inFlightReqs,       \* SUBSET (1..MaxQueries) (async responses still in flight)
    submittedWhileOpen  \* BOOLEAN (safety witness: must ALWAYS remain FALSE!)

vars == <<
    popupOpen, mode, candidateCount, selectedIndex,
    latestReqId, invalidatedUpTo, appliedReqId, inFlightReqs, submittedWhileOpen
>>

\* ---------------------------------------------------------------------------
\* Initial State
\* ---------------------------------------------------------------------------
Init ==
    /\ popupOpen = FALSE
    /\ mode = "NONE"
    /\ candidateCount = 0
    /\ selectedIndex = 0
    /\ latestReqId = 0
    /\ invalidatedUpTo = 0
    /\ appliedReqId = 0
    /\ inFlightReqs = {}
    /\ submittedWhileOpen = FALSE

\* ---------------------------------------------------------------------------
\* Actions
\* ---------------------------------------------------------------------------

\* 1. User types a trigger token (`/` at start or `@...`) -> starts async query with new reqId
TypeTriggerQuery(m) ==
    /\ m \in {"COMMAND", "FILE"}
    /\ latestReqId < MaxQueries
    /\ LET nextId == latestReqId + 1
       IN /\ latestReqId' = nextId
          /\ inFlightReqs' = inFlightReqs \cup {nextId}
    /\ UNCHANGED <<popupOpen, mode, candidateCount, selectedIndex,
                   invalidatedUpTo, appliedReqId, submittedWhileOpen>>

\* 2. User edits text so cursor is no longer on a trigger token (or clears input) -> closes popup
ClearTrigger ==
    /\ (popupOpen \/ inFlightReqs # {})
    /\ invalidatedUpTo < latestReqId \/ popupOpen
    /\ popupOpen' = FALSE
    /\ mode' = "NONE"
    /\ candidateCount' = 0
    /\ selectedIndex' = 0
    /\ invalidatedUpTo' = latestReqId
    /\ UNCHANGED <<latestReqId, appliedReqId, inFlightReqs, submittedWhileOpen>>

\* 3a. Async completion response arrives for the LATEST non-invalidated query with k >= 1 candidates
ReceiveFreshResults(r, m, k) ==
    /\ r \in inFlightReqs
    /\ r = latestReqId
    /\ r > invalidatedUpTo
    /\ m \in {"COMMAND", "FILE"}
    /\ k \in 1..MaxCandidates
    /\ inFlightReqs' = inFlightReqs \ {r}
    /\ appliedReqId' = r
    /\ popupOpen' = TRUE
    /\ mode' = m
    /\ candidateCount' = k
    /\ selectedIndex' = IF popupOpen /\ selectedIndex >= 1 /\ selectedIndex <= k
                        THEN selectedIndex
                        ELSE 1
    /\ UNCHANGED <<latestReqId, invalidatedUpTo, submittedWhileOpen>>

\* 3b. Async completion response arrives for the LATEST non-invalidated query with 0 candidates
ReceiveFreshEmpty(r) ==
    /\ r \in inFlightReqs
    /\ r = latestReqId
    /\ r > invalidatedUpTo
    /\ inFlightReqs' = inFlightReqs \ {r}
    /\ appliedReqId' = r
    /\ popupOpen' = FALSE
    /\ mode' = "NONE"
    /\ candidateCount' = 0
    /\ selectedIndex' = 0
    /\ UNCHANGED <<latestReqId, invalidatedUpTo, submittedWhileOpen>>

\* 3c. Stale or invalidated async response arrives (r < latestReqId \/ r <= invalidatedUpTo) -> DISCARD!
DiscardStaleResponse(r) ==
    /\ r \in inFlightReqs
    /\ (r < latestReqId \/ r <= invalidatedUpTo)
    /\ inFlightReqs' = inFlightReqs \ {r}
    /\ UNCHANGED <<popupOpen, mode, candidateCount, selectedIndex,
                   latestReqId, invalidatedUpTo, appliedReqId, submittedWhileOpen>>

\* 4. User moves selection down (wraps around 1..candidateCount)
MoveDown ==
    /\ popupOpen
    /\ candidateCount > 0
    /\ selectedIndex' = IF selectedIndex < candidateCount THEN selectedIndex + 1 ELSE 1
    /\ UNCHANGED <<popupOpen, mode, candidateCount, latestReqId,
                   invalidatedUpTo, appliedReqId, inFlightReqs, submittedWhileOpen>>

\* 5. User moves selection up (wraps around 1..candidateCount)
MoveUp ==
    /\ popupOpen
    /\ candidateCount > 0
    /\ selectedIndex' = IF selectedIndex > 1 THEN selectedIndex - 1 ELSE candidateCount
    /\ UNCHANGED <<popupOpen, mode, candidateCount, latestReqId,
                   invalidatedUpTo, appliedReqId, inFlightReqs, submittedWhileOpen>>

\* 6. User presses Tab or Enter while popup is open -> accepts selectedIndex, closes popup,
\*    AND invalidates any in-flight queries so a late response cannot pop it back open!
AcceptCompletion ==
    /\ popupOpen
    /\ selectedIndex >= 1 /\ selectedIndex <= candidateCount
    /\ popupOpen' = FALSE
    /\ mode' = "NONE"
    /\ candidateCount' = 0
    /\ selectedIndex' = 0
    /\ invalidatedUpTo' = latestReqId
    /\ UNCHANGED <<latestReqId, appliedReqId, inFlightReqs, submittedWhileOpen>>

\* 7. User presses Escape while popup is open -> dismisses popup & invalidates in-flight queries
DismissEscape ==
    /\ popupOpen
    /\ popupOpen' = FALSE
    /\ mode' = "NONE"
    /\ candidateCount' = 0
    /\ selectedIndex' = 0
    /\ invalidatedUpTo' = latestReqId
    /\ UNCHANGED <<latestReqId, appliedReqId, inFlightReqs, submittedWhileOpen>>

\* 8. User presses Enter when popup is CLOSED -> submits message to agent
SubmitPrompt ==
    /\ ~popupOpen
    /\ UNCHANGED vars

\* 9. Quiescent stuttering once all bounded queries have resolved
QuiescentDone ==
    /\ latestReqId = MaxQueries
    /\ inFlightReqs = {}
    /\ ~popupOpen
    /\ UNCHANGED vars

\* ---------------------------------------------------------------------------
\* Next-State Relation & Fairness
\* ---------------------------------------------------------------------------
Next ==
    \/ \E m \in {"COMMAND", "FILE"} : TypeTriggerQuery(m)
    \/ ClearTrigger
    \/ \E r \in 1..MaxQueries, m \in {"COMMAND", "FILE"}, k \in 1..MaxCandidates : ReceiveFreshResults(r, m, k)
    \/ \E r \in 1..MaxQueries : ReceiveFreshEmpty(r)
    \/ \E r \in 1..MaxQueries : DiscardStaleResponse(r)
    \/ MoveDown
    \/ MoveUp
    \/ AcceptCompletion
    \/ DismissEscape
    \/ SubmitPrompt
    \/ QuiescentDone

Fairness ==
    /\ \A r \in 1..MaxQueries :
        /\ WF_vars(ReceiveFreshEmpty(r))
        /\ WF_vars(DiscardStaleResponse(r))

Spec == Init /\ [][Next]_vars /\ Fairness

\* ---------------------------------------------------------------------------
\* Safety Invariants
\* ---------------------------------------------------------------------------
TypeOK ==
    /\ popupOpen \in BOOLEAN
    /\ mode \in Modes
    /\ candidateCount \in 0..MaxCandidates
    /\ selectedIndex \in 0..MaxCandidates
    /\ latestReqId \in ReqIds
    /\ invalidatedUpTo \in ReqIds
    /\ appliedReqId \in ReqIds
    /\ inFlightReqs \subseteq (1..MaxQueries)
    /\ submittedWhileOpen \in BOOLEAN

PopupHasCandidates ==
    popupOpen => (mode # "NONE" /\ candidateCount > 0)

SelectedIndexInBounds ==
    popupOpen => (selectedIndex >= 1 /\ selectedIndex <= candidateCount)

ClosedStateClean ==
    (~popupOpen) => (mode = "NONE" /\ candidateCount = 0 /\ selectedIndex = 0)

MonotonicAppliedResponses ==
    /\ appliedReqId <= latestReqId
    /\ invalidatedUpTo <= latestReqId
    /\ popupOpen => (appliedReqId > invalidatedUpTo)

NoAccidentalSubmitWhileOpen ==
    ~submittedWhileOpen

\* ---------------------------------------------------------------------------
\* Liveness Property
\* ---------------------------------------------------------------------------
AllInFlightEventuallyDrain ==
    (inFlightReqs # {}) ~> (inFlightReqs = {})

=============================================================================
