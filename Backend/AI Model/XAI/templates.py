from typing import Dict, List


def classify_risk(attack_type: str) -> str:
    at = (attack_type or "").lower()
    if any(x in at for x in ["ddos", "dos", "botnet"]):
        return "High"
    if any(x in at for x in ["portscan", "scan", "brute"]):
        return "Medium"
    if "normal" in at:
        return "None"
    return "Medium"


def render_explanation(facts: Dict) -> str:
    """
    facts:
      attack_type: str
      where: str
      top_concepts: List[str]
      risk: Optional[str]
    """
    attack_type = facts.get("attack_type", "suspicious activity")
    where = facts.get("where", "your system")
    top_concepts: List[str] = facts.get("top_concepts", [])
    risk = facts.get("risk") or classify_risk(attack_type)

    # Normal / benign case
    if risk == "None":
        return (
            "What happened:\n"
            "No malicious behavior was detected. Traffic appears normal.\n\n"
            "How we know:\n"
            "The observed network patterns match your usual baseline.\n\n"
            "Recommended action:\n"
            "No immediate action is needed. The system will continue monitoring."
        )

    # Build 'how we know' section from concepts if we have them
    if top_concepts:
        concept_text = ", ".join(
            c.replace("_", " ").lower() for c in top_concepts[:3]
        )
        how = (
            f"Our system observed patterns such as {concept_text}, "
            f"which are strongly associated with this type of threat."
        )
    else:
        how = (
            "Our system observed unusual network behavior that deviates significantly from normal traffic."
        )

    at_low = attack_type.lower()
    if "ddos" in at_low or "dos" in at_low:
        rec = (
            "Block abusive IPs, enable or tighten rate limiting, and review your firewall/CDN protection."
        )
    elif "scan" in at_low:
        rec = (
            "Block the scanning source addresses and ensure unnecessary ports and services are closed."
        )
    elif "brute" in at_low:
        rec = (
            "Lock or monitor targeted accounts, enforce strong passwords, and enable multi-factor authentication."
        )
    elif "web" in at_low:
        rec = (
            "Review web server logs, patch vulnerable endpoints, and enable a web application firewall (WAF)."
        )
    else:
        rec = (
            "Review logs around this activity and, if confirmed malicious, block the source and tighten access controls."
        )

    return (
        f"What happened:\n"
        f"We detected a likely {attack_type} targeting {where}.\n\n"
        f"How we know:\n"
        f"{how}\n\n"
        f"Risk:\n"
        f"{risk}.\n\n"
        f"Recommended action:\n"
        f"{rec}"
    )
