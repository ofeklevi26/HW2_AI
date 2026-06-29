from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time


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
def ordered_operators(operators):
    ordered = []
    for op in TIE_BREAKING_ORDER:
        if op in operators:
            ordered.append(op)

    return ordered

def expectimax_action_weight(op):
    if op in EXPECTIMAX_ACTION_WEIGHTS:
        return EXPECTIMAX_ACTION_WEIGHTS[op]
    return 1

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
    energy_risk = battery_shortage + 1.5 * charge_penalty

    # Final heuristic:
    # h(s,r) = ScoreDiff + 1.5*PackageUtility - 2*EnergyRisk
    return (score_diff + 1.5* package_utility - 2 * energy_risk
    )


# TODO: section b : fixed-depth helper for deterministic grading
def minimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited minimax.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    def minimax_value(state, current_robot_id, remaining_depth):
        # Course RB-Minimax base case:
        # if terminal state OR depth limit reached, evaluate with h.
        if state.done() or remaining_depth <= 0:
            return heuristic_fn(state, robot_id)

        operators = ordered_operators(state.get_legal_operators(current_robot_id))

        if len(operators) == 0:
            return heuristic_fn(state, robot_id)

        next_robot_id = (current_robot_id + 1) % 2
        # MAX node: our robot's turn
        if current_robot_id == robot_id:
            best_value = float("-inf")

            for op in operators:
                child = state.clone()
                child.apply_operator(current_robot_id, op)

                value = minimax_value(
                    child,
                    next_robot_id,
                    remaining_depth - 1
                )
                best_value = max(best_value, value)

            return best_value

        # MIN node: opponent's turn
        else:
            best_value = float("inf")

            for op in operators:
                child = state.clone()
                child.apply_operator(current_robot_id, op)

                value = minimax_value(
                    child,
                    next_robot_id,
                    remaining_depth - 1
                )

                best_value = min(best_value, value)

            return best_value

    operators = ordered_operators(env.get_legal_operators(robot_id))

    if len(operators) == 0:
        return None

    # If depth is 0, no child search is possible.
    # All actions are equivalent, so return the first by tie-breaking order.
    if depth <= 0 or env.done():
        return operators[0]
    best_operator = operators[0]
    best_value = float("-inf")

    for op in operators:
        child = env.clone()
        child.apply_operator(robot_id, op)

        value = minimax_value(
            child,
            (robot_id + 1) % 2,
            depth - 1
        )

        # Use >, not >=, so ties keep the first action according to TIE_BREAKING_ORDER.
        if value > best_value:
            best_value = value
            best_operator = op

    return best_operator

# TODO: section c : fixed-depth helper for deterministic grading
def alphabeta_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited alpha-beta pruning.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    def alphabeta_value(state, current_robot_id, remaining_depth, alpha, beta):
        # Course RB-AlphaBeta base case:
        # if terminal state OR depth limit reached, evaluate with h.
        if state.done() or remaining_depth <= 0:
            return heuristic_fn(state, robot_id)

        operators = ordered_operators(state.get_legal_operators(current_robot_id))

        if len(operators) == 0:
            return heuristic_fn(state, robot_id)

        next_robot_id = (current_robot_id + 1) % 2

        # MAX node: our robot's turn
        if current_robot_id == robot_id:
            cur_max = float("-inf")

            for op in operators:
                child = state.clone()
                child.apply_operator(current_robot_id, op)

                value = alphabeta_value(
                    child,
                    next_robot_id,
                    remaining_depth - 1,
                    alpha,
                    beta
                )

                cur_max = max(cur_max, value)
                alpha = max(alpha, cur_max)

                # Prune: MIN already has a better option above.
                if cur_max >= beta:
                    return cur_max

            return cur_max

        # MIN node: opponent's turn
        else:
            cur_min = float("inf")

            for op in operators:
                child = state.clone()
                child.apply_operator(current_robot_id, op)

                value = alphabeta_value(
                    child,
                    next_robot_id,
                    remaining_depth - 1,
                    alpha,
                    beta
                )

                cur_min = min(cur_min, value)
                beta = min(beta, cur_min)

                # Prune: MAX already has a better option above.
                if cur_min <= alpha:
                    return cur_min

            return cur_min

    operators = ordered_operators(env.get_legal_operators(robot_id))

    if len(operators) == 0:
        return None

    # If depth is 0 or env is already done, no child search is possible.
    # All actions are equivalent, so return the first by tie-breaking order.
    if depth <= 0 or env.done():
        return operators[0]

    best_operator = operators[0]
    best_value = float("-inf")

    alpha = float("-inf")
    beta = float("inf")

    for op in operators:
        child = env.clone()
        child.apply_operator(robot_id, op)

        value = alphabeta_value(
            child,
            (robot_id + 1) % 2,
            depth - 1,
            alpha,
            beta
        )

        # Use >, not >=, so ties keep the first action according to TIE_BREAKING_ORDER.
        if value > best_value:
            best_value = value
            best_operator = op

        # Root is a MAX choice, so update alpha after each candidate action.
        alpha = max(alpha, best_value)

    return best_operator


# TODO: section d : fixed-depth helper for deterministic grading
def expectimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited expectimax.
    The opponent's legal actions are weighted by EXPECTIMAX_ACTION_WEIGHTS;
    every legal action not in the dictionary has weight 1.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    def expectimax_value(state, current_robot_id, remaining_depth):
        # RB-Expectimax base case:
        # terminal state OR depth limit reached => evaluate with h.
        if state.done() or remaining_depth <= 0:
            return heuristic_fn(state, robot_id)

        operators = ordered_operators(state.get_legal_operators(current_robot_id))

        if len(operators) == 0:
            return heuristic_fn(state, robot_id)

        next_robot_id = (current_robot_id + 1) % 2

        # MAX node: our robot chooses the best action.
        if current_robot_id == robot_id:
            best_value = float("-inf")

            for op in operators:
                child = state.clone()
                child.apply_operator(current_robot_id, op)

                value = expectimax_value(
                    child,
                    next_robot_id,
                    remaining_depth - 1
                )

                best_value = max(best_value, value)

            return best_value

        # Chance node: opponent follows the given probabilistic policy.
        else:
            total_weight = 0
            for op in operators:
                total_weight += expectimax_action_weight(op)

            expected_value = 0

            for op in operators:
                child = state.clone()
                child.apply_operator(current_robot_id, op)

                value = expectimax_value(
                    child,
                    next_robot_id,
                    remaining_depth - 1
                )

                probability = expectimax_action_weight(op) / total_weight
                expected_value += probability * value

            return expected_value

    operators = ordered_operators(env.get_legal_operators(robot_id))

    if len(operators) == 0:
        return None

    # If depth is 0 or env is already done, no child search is possible.
    # All actions are equivalent, so return the first by tie-breaking order.
    if depth <= 0 or env.done():
        return operators[0]

    best_operator = operators[0]
    best_value = float("-inf")

    for op in operators:
        child = env.clone()
        child.apply_operator(robot_id, op)

        value = expectimax_value(
            child,
            (robot_id + 1) % 2,
            depth - 1
        )

        # Use >, not >=, so ties keep the first action according to TIE_BREAKING_ORDER.
        if value > best_value:
            best_value = value
            best_operator = op

    return best_operator


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 4
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        deadline = time.time() + 0.95 * time_limit

        legal_ops = ordered_operators(env.get_legal_operators(agent_id))

        if len(legal_ops) == 0:
            return None

        best_action = legal_ops[0]
        depth = 1

        def timed_minimax_value(state, current_robot_id, remaining_depth):
            if time.time() >= deadline:
                raise TimeoutError()

            # Course RB-Minimax base case:
            # terminal state OR depth limit => evaluate with smart_heuristic.
            if state.done() or remaining_depth <= 0:
                return smart_heuristic(state, agent_id)

            operators = ordered_operators(state.get_legal_operators(current_robot_id))

            if len(operators) == 0:
                return smart_heuristic(state, agent_id)

            next_robot_id = (current_robot_id + 1) % 2

            # MAX node: our robot
            if current_robot_id == agent_id:
                best_value = float("-inf")

                for op in operators:
                    child = state.clone()
                    child.apply_operator(current_robot_id, op)

                    value = timed_minimax_value(
                        child,
                        next_robot_id,
                        remaining_depth - 1
                    )

                    best_value = max(best_value, value)

                return best_value

            # MIN node: opponent robot
            else:
                best_value = float("inf")

                for op in operators:
                    child = state.clone()
                    child.apply_operator(current_robot_id, op)

                    value = timed_minimax_value(
                        child,
                        next_robot_id,
                        remaining_depth - 1
                    )

                    best_value = min(best_value, value)

                return best_value

        # Time-limited minimax using iterative deepening.
        # Keep the best action from the last fully completed depth.
        while True:
            try:
                current_best_action = legal_ops[0]
                current_best_value = float("-inf")

                for op in legal_ops:
                    if time.time() >= deadline:
                        raise TimeoutError()

                    child = env.clone()
                    child.apply_operator(agent_id, op)

                    value = timed_minimax_value(
                        child,
                        (agent_id + 1) % 2,
                        depth - 1
                    )

                    # Tie-breaking: keep first action in TIE_BREAKING_ORDER.
                    if value > current_best_value:
                        current_best_value = value
                        current_best_action = op

                # Only update after a full depth finished.
                best_action = current_best_action
                depth += 1

            except TimeoutError:
                return best_action


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        deadline = time.time() + 0.95 * time_limit

        legal_ops = ordered_operators(env.get_legal_operators(agent_id))

        if len(legal_ops) == 0:
            return None

        best_action = legal_ops[0]
        depth = 1

        def timed_alphabeta_value(state, current_robot_id, remaining_depth, alpha, beta):
            if time.time() >= deadline:
                raise TimeoutError()

            # Course RB-AlphaBeta base case:
            # terminal state OR depth limit => evaluate with smart_heuristic.
            if state.done() or remaining_depth <= 0:
                return smart_heuristic(state, agent_id)

            operators = ordered_operators(state.get_legal_operators(current_robot_id))

            if len(operators) == 0:
                return smart_heuristic(state, agent_id)

            next_robot_id = (current_robot_id + 1) % 2

            # MAX node: our robot
            if current_robot_id == agent_id:
                cur_max = float("-inf")

                for op in operators:
                    if time.time() >= deadline:
                        raise TimeoutError()

                    child = state.clone()
                    child.apply_operator(current_robot_id, op)

                    value = timed_alphabeta_value(
                        child,
                        next_robot_id,
                        remaining_depth - 1,
                        alpha,
                        beta
                    )

                    cur_max = max(cur_max, value)
                    alpha = max(alpha, cur_max)

                    if cur_max >= beta:
                        return cur_max

                return cur_max

            # MIN node: opponent robot
            else:
                cur_min = float("inf")

                for op in operators:
                    if time.time() >= deadline:
                        raise TimeoutError()

                    child = state.clone()
                    child.apply_operator(current_robot_id, op)

                    value = timed_alphabeta_value(
                        child,
                        next_robot_id,
                        remaining_depth - 1,
                        alpha,
                        beta
                    )

                    cur_min = min(cur_min, value)
                    beta = min(beta, cur_min)

                    if cur_min <= alpha:
                        return cur_min

                return cur_min

        # Time-limited alpha-beta using iterative deepening.
        # Keep the best action from the last fully completed depth.
        while True:
            try:
                current_best_action = legal_ops[0]
                current_best_value = float("-inf")

                alpha = float("-inf")
                beta = float("inf")

                for op in legal_ops:
                    if time.time() >= deadline:
                        raise TimeoutError()

                    child = env.clone()
                    child.apply_operator(agent_id, op)

                    value = timed_alphabeta_value(
                        child,
                        (agent_id + 1) % 2,
                        depth - 1,
                        alpha,
                        beta
                    )

                    # Tie-breaking: keep first action in TIE_BREAKING_ORDER.
                    if value > current_best_value:
                        current_best_value = value
                        current_best_action = op

                    # Root is a MAX choice.
                    alpha = max(alpha, current_best_value)

                # Only update after a full depth finished.
                best_action = current_best_action
                depth += 1

            except TimeoutError:
                return best_action


class AgentExpectimax(Agent):
    # TODO: section d : 3
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        deadline = time.time() + 0.95 * time_limit

        legal_ops = ordered_operators(env.get_legal_operators(agent_id))

        if len(legal_ops) == 0:
            return None

        best_action = legal_ops[0]
        depth = 1

        def timed_expectimax_value(state, current_robot_id, remaining_depth):
            if time.time() >= deadline:
                raise TimeoutError()

            # RB-Expectimax base case:
            # terminal state OR depth limit reached => evaluate with smart_heuristic.
            if state.done() or remaining_depth <= 0:
                return smart_heuristic(state, agent_id)

            operators = ordered_operators(state.get_legal_operators(current_robot_id))

            if len(operators) == 0:
                return smart_heuristic(state, agent_id)

            next_robot_id = (current_robot_id + 1) % 2

            # MAX node: our robot
            if current_robot_id == agent_id:
                best_value = float("-inf")

                for op in operators:
                    if time.time() >= deadline:
                        raise TimeoutError()

                    child = state.clone()
                    child.apply_operator(current_robot_id, op)

                    value = timed_expectimax_value(
                        child,
                        next_robot_id,
                        remaining_depth - 1
                    )

                    best_value = max(best_value, value)

                return best_value

            # Chance node: opponent follows weighted probabilistic policy.
            else:
                total_weight = 0
                for op in operators:
                    total_weight += expectimax_action_weight(op)

                expected_value = 0

                for op in operators:
                    if time.time() >= deadline:
                        raise TimeoutError()

                    child = state.clone()
                    child.apply_operator(current_robot_id, op)

                    value = timed_expectimax_value(
                        child,
                        next_robot_id,
                        remaining_depth - 1
                    )

                    probability = expectimax_action_weight(op) / total_weight
                    expected_value += probability * value

                return expected_value

        # Time-limited expectimax using iterative deepening.
        # Keep the best action from the last fully completed depth.
        while True:
            try:
                current_best_action = legal_ops[0]
                current_best_value = float("-inf")

                for op in legal_ops:
                    if time.time() >= deadline:
                        raise TimeoutError()

                    child = env.clone()
                    child.apply_operator(agent_id, op)

                    value = timed_expectimax_value(
                        child,
                        (agent_id + 1) % 2,
                        depth - 1
                    )

                    # Tie-breaking: keep first action in TIE_BREAKING_ORDER.
                    if value > current_best_value:
                        current_best_value = value
                        current_best_action = op

                # Only update after a full depth finished.
                best_action = current_best_action
                depth += 1

            except TimeoutError:
                return best_action


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
