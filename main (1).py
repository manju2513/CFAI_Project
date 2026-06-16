import heapq
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

ROWS, COLS = 20, 26
BATTERY_USAGE_PER_CELL = 2

OBSTACLES = {
    (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
    (8, 5), (9, 5),
    (5, 8), (6, 8), (7, 8),
    (2, 11), (3, 11), (4, 11), (5, 11), (6, 11), (7, 11),
    (2, 15), (3, 15),
    (8, 15), (9, 15), (10, 15), (11, 15), (12, 15),
    (2, 19), (3, 19), (4, 19),
    (10, 18), (11, 18), (12, 18), (13, 18), (14, 18)
}

WAREHOUSE = (10, 1)

BATTERY_STATIONS = {
    1: (16, 10),
    2: (4, 22),
    3: (17, 22)
}

DELIVERY_POINTS = {
    1: (14, 23),
    2: (12, 20),
    3: (11, 20),
    4: (6, 15)
}
# ============================================================
# CO1: PEAS DESCRIPTION
# ============================================================

PEAS = {
    "Performance Measure": [
        "Shortest delivery path",
        "Minimum nodes explored",
        "Safe battery usage",
        "Successful delivery completion",
        "Recharge before battery failure"
    ],

    "Environment": [
        "20 x 26 grid map",
        "Obstacles",
        "Warehouse",
        "Delivery points",
        "Battery stations"
    ],

    "Actuators": [
        "Move up",
        "Move down",
        "Move left",
        "Move right",
        "Recharge battery"
    ],

    "Sensors": [
        "Current position sensor",
        "Battery level sensor",
        "Obstacle detection sensor",
        "Delivery location sensor",
        "Weather risk estimation"
    ]
}

def display_peas():
    print("\n════════ PEAS DESCRIPTION ════════")

    for key, values in PEAS.items():
        print(f"\n{key}:")
        for value in values:
            print(f" - {value}")

    print("══════════════════════════════════")

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def valid(node):
    r, c = node
    return 0 <= r < ROWS and 0 <= c < COLS and node not in OBSTACLES

def neighbors(node):
    r, c = node
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nxt = (r + dr, c + dc)
        if valid(nxt):
            yield nxt

def reconstruct(parent, goal):
    if goal not in parent:
        return []
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]

def bfs(start, goal):
    queue = deque([start])
    parent = {start: None}
    nodes_explored = 0
    while queue:
        current = queue.popleft()
        nodes_explored += 1
        if current == goal:
            break
        for nxt in neighbors(current):
            if nxt not in parent:
                parent[nxt] = current
                queue.append(nxt)
    return reconstruct(parent, goal), nodes_explored

def astar(start, goal):
    pq = [(0, start)]
    g_cost = {start: 0}
    parent = {start: None}
    nodes_explored = 0
    while pq:
        _, current = heapq.heappop(pq)
        nodes_explored += 1
        if current == goal:
            break
        for nxt in neighbors(current):
            new_cost = g_cost[current] + 1
            if nxt not in g_cost or new_cost < g_cost[nxt]:
                g_cost[nxt] = new_cost
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(pq, (priority, nxt))
                parent[nxt] = current
    return reconstruct(parent, goal), nodes_explored

delivery_slots = {}

def is_valid_assignment(point, slot):
    return slot not in delivery_slots.values()

def backtracking_schedule(points, idx=0):
    if idx == len(points):
        return True
    point = points[idx]
    for slot in range(1, len(points) + 1):
        if is_valid_assignment(point, slot):
            delivery_slots[point] = slot
            if backtracking_schedule(points, idx + 1):
                return True
            del delivery_slots[point]
    return False

def bayes_weather_risk():
    P_Rain = 0.30
    P_Delay_given_Rain = 0.80
    P_Delay_given_NoRain = 0.20
    P_Delay = (
        P_Delay_given_Rain * P_Rain
        + P_Delay_given_NoRain * (1 - P_Rain)
    )
    return (P_Delay_given_Rain * P_Rain) / P_Delay

def path_cost(path):
    return sum(1 + random.uniform(0, 0.2) for _ in path)

def utility(path):
    weather_risk = bayes_weather_risk()
    distance_score = len(path)
    risk_penalty = weather_risk * 10
    cost_penalty = path_cost(path)
    return distance_score - cost_penalty - risk_penalty

def create_info_box(name, delivery_id, current, goal, path_len, score, battery, total_distance, status):
    return (
        f"{name} CURRENT DELIVERY: Delivery {delivery_id}\n\n"
        f"Current Position : {current}\n"
        f"Target Location  : {goal}\n\n"
        f"Path Length      : {path_len}\n"
        f"Total Distance   : {total_distance} Cells\n"
        f"Utility Score    : {score:.2f}\n\n"
        f"Battery          : {battery}%\n"
        f"Weather Risk     : {bayes_weather_risk() * 100:.0f}%\n\n"
        f"Next Action      : {status}\n"
        f"Algorithm Used   : {name}"
    )

def create_full_path_with_battery(search_function, name):
    current = WAREHOUSE
    battery = 100
    total_distance = 0
    total_nodes_explored = 0
    delivery_distances = {}

    full_path = [current]
    battery_log = [battery]
    info_log = ["Starting from Warehouse"]
    # NEW: track cumulative nodes explored at each step
    nodes_log = [0]

    print(f"\n🚁 {name} PATH FOLLOWED")

    for delivery_id, goal in DELIVERY_POINTS.items():
        selected_path, nodes_explored = search_function(current, goal)
        total_nodes_explored += nodes_explored
        delivery_distances[delivery_id] = len(selected_path) - 1
        score = utility(selected_path)

        print(f"\n📍 Delivery {delivery_id}")
        print(f"{name} | Length={len(selected_path)} | Score={score:.2f}")

        for cell in selected_path[1:]:
            battery -= BATTERY_USAGE_PER_CELL
            total_distance += 1
            current = cell

            full_path.append(current)
            battery_log.append(battery)
            nodes_log.append(total_nodes_explored)

            current_info = create_info_box(
                name, delivery_id, current, goal,
                len(selected_path), score, battery, total_distance,
                f"Moving to Delivery {delivery_id}"
            )
            info_log.append(current_info)
            print(f"Step {len(full_path)}: {current} | Battery = {battery}%")

        print(f"✅ Delivery {delivery_id} completed at {goal} | Battery = {battery}%")

        if delivery_id == 3:
            station_id = 2
            station_pos = BATTERY_STATIONS[station_id]
            print(f"\n🔋 After Delivery 3, going to Battery Station 2: {station_pos}")

            station_path, station_nodes_explored = search_function(current, station_pos)
            total_nodes_explored += station_nodes_explored

            for cell in station_path[1:]:
                battery -= BATTERY_USAGE_PER_CELL
                total_distance += 1
                current = cell

                full_path.append(current)
                battery_log.append(battery)
                nodes_log.append(total_nodes_explored)

                recharge_info = (
                    f"{name} RECHARGE MODE\n\n"
                    f"Current Position : {current}\n"
                    f"Target Location  : Battery Station 2\n\n"
                    f"Path Length      : {len(station_path)}\n"
                    f"Total Distance   : {total_distance} Cells\n"
                    f"Utility Score    : {utility(station_path):.2f}\n\n"
                    f"Battery          : {battery}%\n"
                    f"Weather Risk     : {bayes_weather_risk() * 100:.0f}%\n\n"
                    f"Next Action      : Going to recharge\n"
                    f"Algorithm Used   : {name}"
                )
                info_log.append(recharge_info)
                print(f"Step {len(full_path)}: {current} | Battery = {battery}%")

            print("⚡ Recharged to 100% at Battery Station 2")
            battery = 100
            battery_log[-1] = battery

    return full_path, battery_log, info_log, total_distance, total_nodes_explored, delivery_distances, nodes_log

def draw_base(ax, title):
    ax.clear()
    ax.set_xlim(0, COLS)
    ax.set_ylim(-1, ROWS)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)

    for r, c in OBSTACLES:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color="black"))

    wr, wc = WAREHOUSE
    ax.add_patch(plt.Circle((wc + 0.5, wr + 0.5), 0.4, color="blue"))
    ax.text(wc + 0.5, wr + 0.5, "W", color="white", ha="center", va="center")

    for station_id, (br, bc) in BATTERY_STATIONS.items():
        ax.add_patch(plt.Circle((bc + 0.5, br + 0.5), 0.4, color="orange"))
        ax.text(bc + 0.5, br + 0.5, f"B{station_id}", color="black", ha="center", va="center", fontsize=8)

    for k, (r, c) in DELIVERY_POINTS.items():
        ax.add_patch(plt.Circle((c + 0.5, r + 0.5), 0.4, color="red"))
        ax.text(c + 0.5, r + 0.5, str(k), color="white", ha="center", va="center")

def draw_path(ax, title, path, step, battery):
    draw_base(ax, title)

    for j in range(step + 1):
        r, c = path[j]
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color="green", alpha=0.4))

    r, c = path[step]
    ax.add_patch(plt.Circle((c + 0.5, r + 0.5), 0.3, color="lime"))

    ax.text(
        1, -0.4,
        f"Step: {step + 1}/{len(path)}     Battery: {battery}%",
        fontsize=11, color="black", fontweight="bold"
    )

def draw_nodes_explored_comparison(ax, bfs_nodes_now, astar_nodes_now):
    """Live-updating bar chart showing nodes explored SO FAR at current step."""
    ax.clear()
    algorithms = ["BFS", "A*"]
    values = [bfs_nodes_now, astar_nodes_now]
    colors = ["#1f77b4", "#ff7f0e"]

    bars = ax.bar(algorithms, values, color=colors)
    ax.set_title("Nodes Explored (Live)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Nodes Explored", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)

    # Dynamic y-axis — leave 20% headroom above the max final value
    max_val = max(bfs_nodes_now, astar_nodes_now, 1)
    ax.set_ylim(0, max_val * 1.25)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.02,
            str(val),
            ha="center", va="bottom", fontsize=8, fontweight="bold"
        )

def animate_both_paths(
    bfs_path, bfs_battery, bfs_info, bfs_nodes_log,
    astar_path, astar_battery, astar_info, astar_nodes_log,
    bfs_nodes_explored, astar_nodes_explored,
    bfs_delivery_distances, astar_delivery_distances
):
    fig = plt.figure(figsize=(18, 14))

    ax_button = fig.add_axes([0.45, 0.97, 0.10, 0.025])
    pause_button = Button(ax_button, "Pause")

    ax_bfs   = fig.add_axes([0.03, 0.40, 0.46, 0.54])
    ax_astar = fig.add_axes([0.51, 0.40, 0.46, 0.54])

    ax_bfs_text          = fig.add_axes([0.03, 0.04, 0.28, 0.30])
    ax_nodes_comparison  = fig.add_axes([0.36, 0.06, 0.28, 0.26])
    ax_astar_text        = fig.add_axes([0.69, 0.04, 0.28, 0.30])

    max_steps = max(len(bfs_path), len(astar_path))
    state = {"paused": False}

    def toggle_pause(event):
        state["paused"] = not state["paused"]
        pause_button.label.set_text("Play" if state["paused"] else "Pause")
        fig.canvas.draw_idle()

    pause_button.on_clicked(toggle_pause)

    for i in range(max_steps):
        bfs_step   = min(i, len(bfs_path) - 1)
        astar_step = min(i, len(astar_path) - 1)

        draw_path(ax_bfs,   "BFS Animated Path", bfs_path,   bfs_step,   bfs_battery[bfs_step])
        draw_path(ax_astar, "A* Animated Path",  astar_path, astar_step, astar_battery[astar_step])

        ax_bfs_text.clear()
        ax_bfs_text.axis("off")
        ax_bfs_text.text(
            0.02, 0.98, bfs_info[bfs_step],
            fontsize=9, family="monospace", va="top",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5")
        )

        ax_astar_text.clear()
        ax_astar_text.axis("off")
        ax_astar_text.text(
            0.02, 0.98, astar_info[astar_step],
            fontsize=9, family="monospace", va="top",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5")
        )

        # ── KEY CHANGE: use per-step node counts from logs ──
        bfs_nodes_now   = bfs_nodes_log[bfs_step]
        astar_nodes_now = astar_nodes_log[astar_step]
        draw_nodes_explored_comparison(ax_nodes_comparison, bfs_nodes_now, astar_nodes_now)

        plt.pause(0.08)

        while state["paused"]:
            plt.pause(0.1)

    plt.show()


# ─── Main ───────────────────────────────────────────────────────────────────

print("\n🚁 SMART AI DRONE DELIVERY SYSTEM")
print("CO1–CO6 INTEGRATED PROJECT")
print("Battery Rule: 1 cell = 2% battery")
display_peas()

backtracking_schedule(list(DELIVERY_POINTS.keys()))

print("\n📅 DELIVERY SCHEDULE")
for point, slot in sorted(delivery_slots.items(), key=lambda x: x[1]):
    print(f"Delivery {point} → Slot {slot}")

print(f"\n🌦 Weather Risk using Bayes: {bayes_weather_risk():.2f}")

print("\n🔍 BFS ROUTE")
bfs_path, bfs_battery, bfs_info, bfs_total_distance, bfs_nodes_explored, bfs_delivery_distances, bfs_nodes_log = \
    create_full_path_with_battery(bfs, "BFS")

print("\n⭐ A* ROUTE")
astar_path, astar_battery, astar_info, astar_total_distance, astar_nodes_explored, astar_delivery_distances, astar_nodes_log = \
    create_full_path_with_battery(astar, "A*")

animate_both_paths(
    bfs_path, bfs_battery, bfs_info, bfs_nodes_log,
    astar_path, astar_battery, astar_info, astar_nodes_log,
    bfs_nodes_explored, astar_nodes_explored,
    bfs_delivery_distances, astar_delivery_distances
)

print("\n════════ FINAL REPORT ════════")
print("CO1 : Agent + Environment ✓")
print("CO2 : BFS + A* Search ✓")
print("CO3 : CSP Scheduling using Backtracking ✓")
print("CO4 : Utility Function ✓")
print("CO5 : Bayesian Inference ✓")
print("CO6 : Hybrid AI Decision System ✓")

print(f"\nBFS Total Steps      : {len(bfs_path)}")
print(f"BFS Total Distance   : {bfs_total_distance} Cells")
print(f"BFS Final Battery    : {bfs_battery[-1]}%")
print(f"BFS Nodes Explored   : {bfs_nodes_explored}")

print(f"\nA* Total Steps       : {len(astar_path)}")
print(f"A* Total Distance    : {astar_total_distance} Cells")
print(f"A* Final Battery     : {astar_battery[-1]}%")
print(f"A* Nodes Explored    : {astar_nodes_explored}")

print("\nBattery Rule  : 1 cell = 2% battery")
print("Recharge Rule : After Delivery 3, drone goes to Battery Station 2 and recharges.")
print("\nIntegrated AI System Successfully Executed")
print("══════════════════════════════")