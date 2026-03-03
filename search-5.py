"""
search.py  (CISC440 minimal search kit)

What students need:
- Subclass Problem: implement actions(state), result(state, action), goal_test(state)
- Optionally implement h(node) for A* / Greedy (NOTE: h takes a *Node*, not a state)

Algorithms included:
- bfs_graph_search(problem)
- ucs(problem)                      # Uniform-Cost
- greedy_best_first_search(problem) # Greedy using h(node)
- astar_search(problem)             # A* using g + h(node)

All UCS/Greedy/A* share the same engine: best_first_graph_search(problem, f).

Designed to be small, readable, and notebook-friendly (no external deps).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import heapq
import sys


def _trace_enabled(problem):
    return getattr(problem, "trace", False)

def _trace_frontier_states(frontier, problem=None, limit=15):
    # PriorityQueue case: show priorities and h-values
    if hasattr(frontier, "peek_entries"):
        out = []
        for prio, node in frontier.peek_entries(limit=limit):
            h = float(problem.h(node)) if (problem and hasattr(problem, "h")) else 0.0
            out.append(f"{node.state}(h={h:.1f}, key={prio:.1f})")
        return out

    # fallback for list/queue-like frontiers
    if hasattr(frontier, "items"):
        return [node.state for _, _, node in frontier.items[:limit]]

    if isinstance(frontier, (list, deque)):
        return [n.state for n in list(frontier)[:limit]]

    return ["<unknown frontier type>"]

def _trace_step(problem, label, node, frontier, explored):
    if not _trace_enabled(problem):
        return

    algo = getattr(problem, "_algo", "Search")

    # Compute h and f in a way that matches the algorithm
    h = float(problem.h(node)) if hasattr(problem, "h") else 0.0
    g = float(node.path_cost)

    if algo == "UCS":
        f = g
    elif algo == "Greedy":
        f = h
    elif algo == "A*":
        f = g + h
    else:
        f = g  # safe default

    print(f"\n[{label}] ALGO={algo}  expand={node.state}  g={g:.1f}  h={h:.1f}  f={f:.1f}  depth={node.depth}")
    print(f"  CLOSED size={len(explored)}  CLOSED={list(explored)[:15]}{'...' if len(explored)>15 else ''}")

    fs = _trace_frontier_states(frontier, problem=problem)
    frontier_kind = "PriorityQueue" if hasattr(frontier, "items") else type(frontier).__name__
    print(f"  OPEN({frontier_kind}) size={len(fs)}  OPEN={fs[:15]}{'...' if len(fs)>15 else ''}")

# ---------------------------------------------------------------------
# 1) Problem definition (students implement a subclass)
# ---------------------------------------------------------------------

class Problem:
    """
    To define a new search problem, subclass Problem and implement:
      - actions(state) -> iterable of actions
      - result(state, action) -> next_state
      - goal_test(state) -> bool
    Optionally implement:
      - path_cost(c, state1, action, state2) -> new_cost
      - h(node) -> heuristic estimate to goal (used by Greedy/A*)
    """

    def __init__(self, initial: Any, goal: Any = None):
        self.initial = initial
        self.goal = goal

    def actions(self, state: Any) -> Iterable[Any]:
        raise NotImplementedError

    def result(self, state: Any, action: Any) -> Any:
        raise NotImplementedError

    def goal_test(self, state: Any) -> bool:
        return state == self.goal

    def path_cost(self, c: float, state1: Any, action: Any, state2: Any) -> float:
        # Default: each step costs 1
        return c + 1

    def h(self, node: "Node") -> float:
        # Default: no guidance
        return 0.0


class InstrumentedProblem:
    """
    Wrap a Problem to count calls (useful for classroom comparisons).
    """
    def __init__(self, problem: Problem):
        self.problem = problem
        self.succs = 0
        self.goal_tests = 0
        self.states = 0

    def actions(self, state):
        self.succs += 1
        return self.problem.actions(state)

    def result(self, state, action):
        self.states += 1
        return self.problem.result(state, action)

    def goal_test(self, state):
        self.goal_tests += 1
        return self.problem.goal_test(state)

    def path_cost(self, c, state1, action, state2):
        return self.problem.path_cost(c, state1, action, state2)

    def h(self, node):
        return self.problem.h(node)

    def __getattr__(self, name):
        return getattr(self.problem, name)


# ---------------------------------------------------------------------
# 2) Node (used internally by the search algorithms)
# ---------------------------------------------------------------------

@dataclass(order=False)
class Node:
    state: Any
    parent: Optional["Node"] = None
    action: Any = None
    path_cost: float = 0.0
    depth: int = 0

    '''def expand(self, problem: Problem) -> List["Node"]:
        children: List[Node] = []
        for action in problem.actions(self.state):
            next_state = problem.result(self.state, action)
            new_cost = problem.path_cost(self.path_cost, self.state, action, next_state)
            children.append(Node(
                state=next_state,
                parent=self,
                action=action,
                path_cost=new_cost,
                depth=self.depth + 1,
            ))
        return children
        '''
    def expand(self, problem: Problem) -> List["Node"]:
        children: List[Node] = []
    
        actions = list(problem.actions(self.state))
    
        # ---- TRACE OUTPUT (for all search algorithms) ----
        if getattr(problem, "trace", False):
            algo = getattr(problem, "_algo", "Search")
            h = float(problem.h(self)) if hasattr(problem, "h") else 0.0
            g = float(self.path_cost)
            f = g if algo == "UCS" else (h if algo == "Greedy" else (g + h))
            print(f"\nExpanding state: {self.state} (g={g}, h={h}, f={f}, depth={self.depth})")
            print(f"  Actions: {actions}")
    
        for action in actions:
            next_state = problem.result(self.state, action)
            new_cost = problem.path_cost(self.path_cost, self.state, action, next_state)
    
            if getattr(problem, "trace", False):
                print(f"    -> {action} => {next_state} (new g={new_cost})")
    
            children.append(Node(
                state=next_state,
                parent=self,
                action=action,
                path_cost=new_cost,
                depth=self.depth + 1,
            ))
    
        return children

    def solution(self) -> List[Any]:
        return [n.action for n in self.path()[1:]]

    def path(self) -> List["Node"]:
        node, back = self, []
        while node is not None:
            back.append(node)
            node = node.parent
        return list(reversed(back))

    def __repr__(self):
        return f"Node(state={self.state}, action={self.action}, g={self.path_cost}, depth={self.depth})"

    def pretty(self):
        parent_state = None if self.parent is None else self.parent.state
        return (f"state={self.state}  action={self.action}  "
            f"g={self.path_cost}  depth={self.depth}  parent={parent_state}")



# ---------------------------------------------------------------------
# 3) Priority queue + memoization
# ---------------------------------------------------------------------

class PriorityQueue:
    """A min-priority queue that supports update-by-state."""
    def __init__(self, key: Callable[[Node], float]):
        self.key = key
        self.heap: List[Tuple[float, int, Node]] = []
        self.best_by_state: Dict[Any, float] = {}
        self.counter = 0

    def push_or_update(self, node: Node):
        k = self.key(node)
        s = node.state
        if s in self.best_by_state and self.best_by_state[s] <= k:
            return
        self.best_by_state[s] = k
        self.counter += 1
        heapq.heappush(self.heap, (k, self.counter, node))

    @property
    def items(self):
        """
        For tracing only: return a list of (priority, tie, node) that are still valid
        according to best_by_state.
        """
        out = []
        for k, t, node in self.heap:
            if self.best_by_state.get(node.state) == k:
                out.append((k, t, node))
        # optional: show them in pop order
        out.sort(key=lambda x: (x[0], x[1]))
        return out

    def peek_entries(self, limit=None):
        """
        For tracing: return valid frontier entries as a list of (priority, node),
        sorted in the order they would be popped.
        """
        entries = []
        for k, t, node in self.heap:
            if self.best_by_state.get(node.state) == k:
                entries.append((k, t, node))
        entries.sort(key=lambda x: (x[0], x[1]))
        if limit is not None:
            entries = entries[:limit]
        return [(k, node) for (k, _, node) in entries]

    def pop(self) -> Node:
        while self.heap:
            k, _, node = heapq.heappop(self.heap)
            if self.best_by_state.get(node.state) == k:
                # consume this best entry
                del self.best_by_state[node.state]
                return node
        raise KeyError("pop from empty priority queue")

    def __len__(self) -> int:
        return len(self.best_by_state)


def memoize(fn: Callable[[Node], float]) -> Callable[[Node], float]:
    cache: Dict[int, float] = {}
    def wrapped(node: Node) -> float:
        key = id(node)
        if key not in cache:
            cache[key] = fn(node)
        return cache[key]
    return wrapped





# ---------------------------------------------------------------------
# 5) Algorithms students call (thin wrappers)
# ---------------------------------------------------------------------

def bfs_graph_search(problem: Problem, *, return_metrics: bool = False):
    """BFS using a FIFO queue (graph search)."""
    start = Node(problem.initial)
    frontier: List[Node] = [start]
    explored: Set[Any] = set()

    expanded = 0
    generated = 1

    while frontier:
        node = frontier.pop(0)
        expanded += 1
        _trace_step(problem, "POP", node, frontier, explored)
        if problem.goal_test(node.state):
            metrics = {"expanded": expanded, "generated": generated}
            return (node, metrics) if return_metrics else node

        explored.add(node.state)
        for child in node.expand(problem):
            generated += 1
            if child.state not in explored and all(n.state != child.state for n in frontier):
                frontier.append(child)
        if _trace_enabled(problem):
            print("  (after enqueuing successors)")
            _trace_step(problem, "AFTER", node, frontier, explored)

    metrics = {"expanded": expanded, "generated": generated}
    return (None, metrics) if return_metrics else None



def dfs_graph_search(problem: Problem, *, return_metrics: bool = False):
    """Depth-First Search using a LIFO stack (graph search)."""
    start = Node(problem.initial)
    frontier: List[Node] = [start]   # stack (LIFO)
    explored: Set[Any] = set()

    expanded = 0
    generated = 1

    while frontier:
        node = frontier.pop()        # LIFO pop
        expanded += 1
        _trace_step(problem, "POP", node, frontier, explored)

        if problem.goal_test(node.state):
            metrics = {"expanded": expanded, "generated": generated}
            return (node, metrics) if return_metrics else node

        explored.add(node.state)

        # Reverse children so DFS explores in a natural order
        for child in reversed(node.expand(problem)):
            generated += 1
            if child.state not in explored and all(n.state != child.state for n in frontier):
                frontier.append(child)
            if _trace_enabled(problem):
                print("  (after pushing successors)")
                _trace_step(problem, "AFTER", node, frontier, explored)

    metrics = {"expanded": expanded, "generated": generated}
    return (None, metrics) if return_metrics else None

def depth_limited_graph_search(problem: Problem, limit: int = 50, *, return_metrics: bool = False):
    """
    Depth-Limited GRAPH Search (DFS with a depth bound + explored bookkeeping).

    - Prevents infinite loops via a 'best_depth' map:
        We only expand a state if we reach it at a strictly smaller depth
        than we have seen before.
    - This is more 'graph-search-like' than plain recursive DLS.

    Returns:
        goal_node or 'cutoff' or None
        (optionally with metrics dict if return_metrics=True)
    """
    start = Node(problem.initial)
    frontier: List[Node] = [start]   # stack for DFS
    best_depth: Dict[Any, int] = {start.state: 0}

    expanded = 0
    generated = 1
    cutoff_occurred = False

    while frontier:
        node = frontier.pop()
        expanded += 1

        # Use your existing trace printout style
        _trace_step(problem, "POP", node, frontier, set(best_depth.keys()))

        if problem.goal_test(node.state):
            metrics = {"expanded": expanded, "generated": generated}
            return (node, metrics) if return_metrics else node

        if node.depth >= limit:
            cutoff_occurred = True
            continue

        # Expand successors (DFS style: push in reverse to keep natural order)
        children = node.expand(problem)
        for child in reversed(children):
            generated += 1

            d = child.depth
            s = child.state

            # Only consider this child if we haven't seen s before,
            # OR if we are seeing it at a *smaller* depth than before.
            prev = best_depth.get(s)
            if prev is None or d < prev:
                best_depth[s] = d
                frontier.append(child)

        if _trace_enabled(problem):
            print("  (after pushing successors)")
            _trace_step(problem, "AFTER", node, frontier, set(best_depth.keys()))

    metrics = {"expanded": expanded, "generated": generated, "cutoff": cutoff_occurred}
    result = 'cutoff' if cutoff_occurred else None
    return (result, metrics) if return_metrics else result


def iterative_deepening_graph_search(problem: Problem, max_depth: int = 50, *, return_metrics: bool = False):
    """
    Iterative Deepening using depth-limited GRAPH search.
    Tries depth=0..max_depth.
    """
    total_expanded = 0
    total_generated = 0

    for depth in range(max_depth + 1):
        result = depth_limited_graph_search(problem, limit=depth, return_metrics=True)

        # result is either (Node, metrics) OR ('cutoff'/None, metrics)
        outcome, metrics = result
        total_expanded += metrics.get("expanded", 0)
        total_generated += metrics.get("generated", 0)

        if outcome not in ('cutoff', None):
            if return_metrics:
                metrics_all = {
                    "total_expanded": total_expanded,
                    "total_generated": total_generated,
                    "found_at_depth": depth
                }
                return outcome, metrics_all
            return outcome

        if outcome is None:
            # No solution even without cutoff at this depth
            break

    if return_metrics:
        return None, {"total_expanded": total_expanded, "total_generated": total_generated}
    return None

def dls_graph_search(problem: Problem, limit: int = 50, *, return_metrics: bool = False):
    problem.show_cost = False
    return depth_limited_graph_search(problem, limit=limit, return_metrics=return_metrics)

def iddfs_graph_search(problem: Problem, max_depth: int = 50, *, return_metrics: bool = False):
    
    return iterative_deepening_graph_search(problem, max_depth=max_depth, return_metrics=return_metrics)

# ---------------------------------------------------------------------
# 4) One engine: best-first GRAPH search
# ---------------------------------------------------------------------

def best_first_graph_search(problem: Problem, f: Callable[[Node], float], *, return_metrics: bool = False):
    """
    Generic best-first GRAPH search.
    - f(node): priority score (lower expands sooner)
    Returns goal Node (or (goal Node, metrics) if return_metrics=True).
    """
    f = memoize(f)

    start = Node(problem.initial)
    frontier = PriorityQueue(key=f)
    frontier.push_or_update(start)
    explored: Set[Any] = set()

    expanded = 0
    generated = 1  # includes start

    while len(frontier) > 0:
        node = frontier.pop()
        expanded += 1
        _trace_step(problem, "POP", node, frontier, explored)
        if problem.goal_test(node.state):
            metrics = {"expanded": expanded, "generated": generated}
            return (node, metrics) if return_metrics else node

        explored.add(node.state)

        for child in node.expand(problem):
            generated += 1
            if child.state not in explored:
                frontier.push_or_update(child)
        if _trace_enabled(problem):
            print("  (after pushing successors)")
            _trace_step(problem, "AFTER", node, frontier, explored)

    metrics = {"expanded": expanded, "generated": generated}
    return (None, metrics) if return_metrics else None



 
def ucs(problem: Problem, *, return_metrics: bool = False):
    """Uniform-Cost Search: f(n) = g(n)"""
    problem._algo = "UCS"
    return best_first_graph_search(problem, f=lambda n: n.path_cost, return_metrics=return_metrics)


def greedy_best_first_search(problem: Problem, *, return_metrics: bool = False):
    """Greedy Best-First Search: f(n) = h(n)"""
    problem._algo = "Greedy Best First"
    return best_first_graph_search(problem, f=lambda n: problem.h(n), return_metrics=return_metrics)


def astar_search(problem: Problem, *, return_metrics: bool = False):
    """A* Search: f(n) = g(n) + h(n)"""
    problem._algo = "A* Search"
    return best_first_graph_search(problem, f=lambda n: n.path_cost + problem.h(n), return_metrics=return_metrics)

# ---------------------------------------------------------------------
# 6) Small helper for printing results (nice for demos)
# ---------------------------------------------------------------------

def print_solution(goal_node: Optional[Node], *, metrics: Optional[dict] = None,
                   show_cost: bool = True, problem: Optional[Problem] = None,
                   show_h: bool = False):
    if goal_node is None:
        print("No solution found.")
        if metrics:
            print("Metrics:", metrics)
        return

    actions = goal_node.solution()
    path_nodes = goal_node.path()
    states = [n.state for n in path_nodes]

    print("Solution actions:", actions)
    print("Path states:", states)

    if show_cost:
        print("Total cost g:", goal_node.path_cost)

    # print h values along the solution path (works for Greedy/A*/also fine for BFS)
    if show_h and problem is not None and hasattr(problem, "h"):
        hs = [float(problem.h(n)) for n in path_nodes]
        print("Path h-values:", hs)

    if metrics:
        print("Metrics:", metrics)



class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)


class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf


import math

def distance(a, b):
    """Straight-line (Euclidean) distance between two (x, y) points."""
    (x1, y1) = a
    (x2, y2) = b
    return math.hypot(x1 - x2, y1 - y2)



def compare_searchers(problems, header,
                      searchers=[bfs_graph_search, dfs_graph_search, ucs]):

    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        #goal = searcher(p)
        goal, metrics = searcher(p, return_metrics=True)


        
        final_state = None if goal is None else getattr(goal, "state", goal)
        initial_state = getattr(problem, "initial", None)

        # succs ≈ nodes expanded, states ≈ generated (as you noted)
        #return [p.succs, p.goal_tests, p.states,  initial_state, final_state]
        return [metrics["expanded"], p.goal_tests, metrics["generated"], initial_state, final_state]

    # loop through each problem in the list
    for i, problem in enumerate(problems, start=1):
        print(f"\nProblem {i}: {getattr(problem, 'initial', None)} → {getattr(problem, 'goal', None)}")

        table = [[s.__name__] + do(s, problem) for s in searchers]
        print_table(table, header)




def print_table(table, header):
    # compute column widths
    cols = list(zip(*([header] + table)))
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(row):
        return " | ".join(str(x).ljust(w) for x, w in zip(row, widths))

    print(fmt_row(header))
    print("-+-".join("-" * w for w in widths))
    for row in table:
        print(fmt_row(row))

