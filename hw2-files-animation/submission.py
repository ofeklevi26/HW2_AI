from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


EXPECTIMAX_ACTION_WEIGHTS = {
    "move north": 4,
    "charge": 4,
}

TIE_BREAKING_ORDER = [
    "drop off",
    "pick up",
    "charge",
    "move north",
    "move east",
    "move south",
    "move west",
    "park",
]


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    opponent = env.get_robot((robot_id + 1) % 2)

    # ScoreDiff(s, r)
    score_diff = robot.credit - opponent.credit

    def package_reward(package):
        # Reward(p) = 3 * Manhattan(origin(p), destination(p))
        return 3 * manhattan_distance(package.position, package.destination)

    def route_dist(package):
        # If the robot is carrying a package, it only needs to reach its destination.
        if robot.package is not None:
            return manhattan_distance(robot.position, package.destination)

        # If the robot is not carrying, it needs to reach the package and then deliver it.
        return (
            manhattan_distance(robot.position, package.position)
            + manhattan_distance(package.position, package.destination)
        )

    # Choose p*
    if robot.package is not None:
        selected_package = robot.package
    else:
        active_packages = [
            package for package in env.packages[0:2]
            if package.on_board
        ]

        # Defensive case. In the normal environment there should be active packages,
        # but this avoids a crash if something unusual happens.
        if len(active_packages) == 0:
            return 0.5 * score_diff

        selected_package = max(
            active_packages,
            key=lambda package: package_reward(package) - route_dist(package)
        )

    # PackageUtility(s, r) = Reward(p*) - RouteDist(s, r, p*)
    selected_route_dist = route_dist(selected_package)
    package_utility = package_reward(selected_package) - selected_route_dist

    # BatteryShortage(s, r)
    battery_shortage = max(0, selected_route_dist - robot.battery)

    # ChargePenalty(s, r)
    # The charger is relevant only when there is not enough battery.
    if battery_shortage > 0:
        charge_dist = min(
            manhattan_distance(robot.position, charge_station.position)
            for charge_station in env.charge_stations
        )
        charge_penalty = charge_dist
    else:
        charge_penalty = 0

    # EnergyRisk(s, r)
    energy_risk = battery_shortage + 0.5 * charge_penalty

    # Final heuristic:
    # h(s,r) = 0.5*ScoreDiff + 3*PackageUtility - 4*EnergyRisk
    return (0.5 * score_diff + 3 * package_utility - 4 * energy_risk
    )


# TODO: section b : fixed-depth helper for deterministic grading
def minimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited minimax.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    raise NotImplementedError()


# TODO: section c : fixed-depth helper for deterministic grading
def alphabeta_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited alpha-beta pruning.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    raise NotImplementedError()


# TODO: section d : fixed-depth helper for deterministic grading
def expectimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited expectimax.
    The opponent's legal actions are weighted by EXPECTIMAX_ACTION_WEIGHTS;
    every legal action not in the dictionary has weight 1.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    raise NotImplementedError()


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 4
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 3
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
