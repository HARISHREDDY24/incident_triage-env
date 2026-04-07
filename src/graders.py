def grade_episode(initial_orders, final_orders, step_count, max_steps):
    total_orders = len(initial_orders)
    fulfilled_orders = sum(1 for o in final_orders if o.fulfilled)

    score = fulfilled_orders / total_orders if total_orders else 0.0

    spoiled = sum(1 for o in final_orders if not o.fulfilled and o.days_to_spoil <= 0)
    score -= spoiled * 0.2

    if fulfilled_orders == total_orders:
        efficiency_bonus = ((max_steps - step_count) / max_steps) * 0.1
        score += efficiency_bonus

    return max(0.0, min(1.0, score))