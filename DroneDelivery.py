import heapq
import random
from collections import deque
import matplotlib.pyplot as plt

random.seed(42)

# ============================================================
# CO1: AGENT & ENVIRONMENT MODEL
# ============================================================

ROWS, COLS = 20, 26


OBSTACLES = {
    # Left wall
    (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
    (8, 5), (9, 5),

    # Center-left wall
    (5, 8), (6, 8), (7, 8),

    # Center wall
    (2, 11), (3, 11), (4, 11),
    (5, 11), (6, 11), (7, 11),

    # Upper-right wall
    (2, 15), (3, 15),

    # Middle-right wall
    (8, 15), (9, 15), (10, 15),
    (11, 15), (12, 15),

    # Right-side wall
    (2, 19), (3, 19), (4, 19),

    # Lower-right wall
    (10, 18), (11, 18), (12, 18),
    (13, 18), (14, 18),

    # Extra challenge obstacles
    (9, 10), (9, 11),
    (10, 10),

    (6, 17), (7, 17),

    (8, 20), (9, 20),



    # Obstacles near Delivery 1 route
    (14, 20), (14, 21),
    (13, 20),
    (12, 19), (13, 19),
# Additional obstacles (bottom area)

(15, 4), (16, 4), (17, 4),

(15, 8), (15, 9), (15, 10),

(17, 13), (18, 13),

(16, 21), (17, 21), (18, 21)
}

WAREHOUSE = (10, 1)

DELIVERY_POINTS = {
    1: (14, 23),
    2: (12, 20),
    3: (11, 20),
    4: (6, 15)
}
# ============================================================
# CO1: PEAS MODEL
# ============================================================

PEAS = {
    "Performance": [
        "Fast Delivery Time",
        "Optimal Path",
        "Low Energy Usage",
        "Safe Delivery"
    ],

    "Environment": [
        "City Grid",
        "Warehouse",
        "Delivery Locations",
        "Obstacles / No-Fly Zones"
    ],

    "Actuators": [
        "Move Up",
        "Move Down",
        "Move Left",
        "Move Right",
        "Pick Package",
        "Drop Package"
    ],

    "Sensors": [
        "GPS Position",
        "Obstacle Detector",
        "Weather Monitor",
        "Battery Sensor"
    ]
}


def display_peas():

    print("\n════════ PEAS MODEL ════════")

    for category, items in PEAS.items():

        print(f"\n{category}:")

        for item in items:
            print(f"  • {item}")

    print("════════════════════════════")

# ============================================================
# CO2: SEARCH UTILITIES
# ============================================================

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def valid(node):
    r, c = node

    return (
        0 <= r < ROWS
        and 0 <= c < COLS
        and node not in OBSTACLES
    )


def neighbors(node):

    r, c = node

    directions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ]

    for dr, dc in directions:

        nxt = (r + dr, c + dc)

        if valid(nxt):
            yield nxt


# ============================================================
# PATH RECONSTRUCTION
# ============================================================

def reconstruct(parent, goal):

    if goal not in parent:
        return []

    path = []

    cur = goal

    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    return path[::-1]


# ============================================================
# CO2: BFS
# ============================================================

def bfs(start, goal):

    queue = deque([start])
    expanded = 0

    parent = {
        start: None
    }

    while queue:
        expanded += 1

        current = queue.popleft()

        if current == goal:
            break

        for nxt in neighbors(current):

            if nxt not in parent:

                parent[nxt] = current
                queue.append(nxt)

    return reconstruct(parent, goal), expanded


# ============================================================
# CO2: A* SEARCH
# ============================================================

def astar(start, goal):

    pq = [(0, start)]
    expanded = 0

    g_cost = {
        start: 0
    }

    parent = {
        start: None
    }

    while pq:

        expanded += 1
        _, current = heapq.heappop(pq)

        if current == goal:
            break

        for nxt in neighbors(current):

            new_cost = g_cost[current] + 1

            if (
                nxt not in g_cost
                or new_cost < g_cost[nxt]
            ):

                g_cost[nxt] = new_cost

                priority = (
                    new_cost
                    + heuristic(nxt, goal)
                )

                heapq.heappush(
                    pq,
                    (priority, nxt)
                )

                parent[nxt] = current

    return reconstruct(parent, goal), expanded


# ============================================================
# CO3: CSP SCHEDULING (BACKTRACKING)
# ============================================================

delivery_slots = {}


def is_valid_assignment(point, slot):

    return (
        slot not in delivery_slots.values()
    )


def backtracking_schedule(points, idx=0):

    if idx == len(points):
        return True

    point = points[idx]

    for slot in range(
        1,
        len(points) + 1
    ):

        if is_valid_assignment(
            point,
            slot
        ):

            delivery_slots[point] = slot

            if backtracking_schedule(
                points,
                idx + 1
            ):
                return True

            del delivery_slots[point]

    return False


# ============================================================
# CO5: BAYESIAN INFERENCE
# ============================================================

def bayes_weather_risk():

    P_Rain = 0.30

    P_Delay_given_Rain = 0.80

    P_Delay_given_NoRain = 0.20

    P_Delay = (
        P_Delay_given_Rain * P_Rain
        + P_Delay_given_NoRain * (1 - P_Rain)
    )

    P_Rain_given_Delay = (
        P_Delay_given_Rain * P_Rain
    ) / P_Delay

    return P_Rain_given_Delay


# ============================================================
# PATH COST
# ============================================================

def path_cost(path):
    return len(path)


# ============================================================
# CO4: UTILITY FUNCTION
# ============================================================

def utility(path):

    weather_risk = bayes_weather_risk()

    distance_score = len(path)

    risk_penalty = weather_risk * 10

    cost_penalty = path_cost(path)

    return (
        distance_score
        - cost_penalty
        - risk_penalty
    )


# ============================================================
# CO6: HYBRID AI DECISION SYSTEM
# ============================================================



def evaluate_all(start, goal):

    bfs_path, bfs_expanded = bfs(start, goal)
    astar_path, astar_expanded = astar(start, goal)

    subsection("BFS RESULT")
    print("Path Length :", len(bfs_path))
    print("Nodes Expanded :", bfs_expanded)

    subsection("A* RESULT")
    print("Path Length :", len(astar_path))
    print("Nodes Expanded :", astar_expanded)

    bfs_score = utility(bfs_path)
    astar_score = utility(astar_path)

    subsection("UTILITY ANALYSIS")
    print(f"BFS Score : {bfs_score:.2f}")
    print(f"A* Score  : {astar_score:.2f}")

    if astar_expanded < bfs_expanded:
        print("\nA* explored fewer nodes, so it is more efficient.")
    elif bfs_expanded < astar_expanded:
        print("\nBFS explored fewer nodes.")
    else:
        print("\nBoth algorithms explored the same number of nodes.")

    if astar_score >= bfs_score:
        print("\nSELECTED : A*")
        return astar_path
    else:
        print("\nSELECTED : BFS")
        return bfs_path

# ============================================================
# VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))


def draw_environment():

    ax.clear()

    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)

    ax.invert_yaxis()

    ax.set_aspect("equal")

    ax.set_title(
        "Drone Delivery Route Planner",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.grid(True, alpha=0.3)

    # Obstacles

    for r, c in OBSTACLES:

        ax.add_patch(
            plt.Rectangle(
                (c, r),
                1,
                1,
                color="black"
            )
        )

    # Warehouse

    wr, wc = WAREHOUSE

    ax.add_patch(
        plt.Circle(
            (wc + 0.5, wr + 0.5),
            0.4,
            color="blue"
        )
    )

    ax.text(
        wc + 0.5,
        wr + 0.5,
        "W",
        color="white",
        ha="center",
        va="center"
    )

    # Delivery Points

    for k, (r, c) in DELIVERY_POINTS.items():

        ax.add_patch(
            plt.Circle(
                (c + 0.5, r + 0.5),
                0.4,
                color="red"
            )
        )

        ax.text(
            c + 0.5,
            r + 0.5,
            str(k),
            color="white",
            ha="center",
            va="center"
        )



# ============================================================
# PROFESSIONAL REPORTING
# ============================================================

def section(title):
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)

def subsection(title):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)

def system_report():
    section("DRONE DELIVERY SYSTEM INFORMATION")
    print(f"Warehouse Location : {WAREHOUSE}")
    print(f"Total Deliveries   : {len(DELIVERY_POINTS)}")
    print(f"Grid Size          : {ROWS} x {COLS}")
    print(f"Obstacle Count     : {len(OBSTACLES)}")
    print()
    print("Agent Type         : Goal Based Agent")
    print("Search Algorithms  : BFS, A*")
    print("Decision Model     : Utility Based")
    print("Inference Model    : Bayesian")


# ============================================================
# RUN SYSTEM
# ============================================================

section("SMART AI DRONE DELIVERY SYSTEM")
system_report()
display_peas()

# CO3 Scheduling

backtracking_schedule(
    list(DELIVERY_POINTS.keys())
)

print("\n📅 DELIVERY SCHEDULE")

for point, slot in sorted(
    delivery_slots.items(),
    key=lambda x: x[1]
):

    print(
        f"Delivery {point}"
        f" → Slot {slot}"
    )

section("BAYESIAN WEATHER ANALYSIS")

risk = bayes_weather_risk()

print("\nP(Rain)             = 0.30")
print("P(Delay | Rain)     = 0.80")
print("P(Delay | No Rain)  = 0.20")

print(f"\nP(Rain | Delay)     = {risk:.2f}")

if risk < 0.4:
    level = "LOW"
elif risk < 0.7:
    level = "MEDIUM"
else:
    level = "HIGH"

print(f"\nRisk Level          = {level}")

current = WAREHOUSE

full_path = []
route_history = []

for delivery_id, goal in DELIVERY_POINTS.items():

    print(
        f"\n📍 Delivery {delivery_id}"
    )

    path = evaluate_all(
        current,
        goal
    )

    route_history.append(
        (delivery_id, len(path))
    )

    if full_path:
        full_path.extend(path[1:])
    else:
        full_path.extend(path)

    current = goal

    print(
        f"✅ Selected Path Length = {len(path)}"
    )


# ============================================================
# ANIMATION
# ============================================================

draw_environment()

for i in range(len(full_path)):

    draw_environment()

    for j in range(i):

        r, c = full_path[j]

        ax.add_patch(
            plt.Rectangle(
                (c, r),
                1,
                1,
                color="green",
                alpha=0.4
            )
        )

    r, c = full_path[i]

    ax.add_patch(
        plt.Circle(
            (c + 0.5, r + 0.5),
            0.3,
            color="lime"
        )
    )

    print(
        f"🚁 Step {i+1}/{len(full_path)}"
        f" → {full_path[i]}"
    )

    plt.pause(0.05)

# ============================================================
# FINAL REPORT
# ============================================================



print("\n" + "=" * 60)
print("FINAL DRONE DELIVERY REPORT")
print("=" * 60)

print("\nWarehouse          : (10,1)")

print("\nDeliveries Completed")
print("--------------------")

for delivery in DELIVERY_POINTS:
    print(f"✓ Delivery {delivery}")

print("\nRoute Statistics")
print("----------------")
print(f"Total Steps      : {len(full_path)}")
print(f"Obstacle Count   : {len(OBSTACLES)}")
print(f"Weather Risk     : {bayes_weather_risk():.2f}")

print("\nAlgorithms Used")
print("---------------")
print("✓ BFS")
print("✓ A*")
print("✓ CSP")
print("✓ Bayesian Inference")
print("✓ Utility Function")

print("\nOverall Status")
print("--------------")
print("SUCCESS")

print("\n" + "=" * 60)
print("PROJECT EXECUTED SUCCESSFULLY")
print("=" * 60)


plt.show()
