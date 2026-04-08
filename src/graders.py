def clamp(score):
    # STRICTLY between 0 and 1
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return score

def grade_incident(state, *args, **kwargs):
    # state is usually passed as a dict or object
    # If state is an object, use state.disk_usage
    disk = state.get("disk_usage", 100.0) if isinstance(state, dict) else getattr(state, "disk_usage", 100.0)
    services = state.get("services", {}) if isinstance(state, dict) else getattr(state, "services", {})

    # 1. Disk Score (Lower usage = Higher score)
    disk_score = 1.0 - (disk / 100.0)
    
    # 2. Service Score (Running = 1.0, Stopped/Error = 0.0)
    running_count = sum(1 for s in services.values() if s == "running")
    total_services = len(services) if services else 1
    service_score = running_count / total_services

    # Average them and add tiny smoothing
    final_score = (disk_score + service_score) / 2
    final_score = final_score * 0.98 + 0.01
    
    return clamp(final_score)
