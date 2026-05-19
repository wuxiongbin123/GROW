import copy
import logging
import re


SP_BENCH_PROFILE = "sp_bench"
SMELT_DISTRACTOR_ITEMS = ["charcoal", "stick", "oak_planks", "sand", "clay_ball", "cobblestone"]
CRAFT_DISTRACTOR_ITEMS = [
    "stick",
    "oak_planks",
    "spruce_planks",
    "cobblestone",
    "coal",
    "charcoal",
    "iron_nugget",
    "gold_nugget",
]


def apply_benchmark_profile(task_config: dict, args) -> dict:
    profile = getattr(args, "benchmark_profile", "") or ""
    if profile != SP_BENCH_PROFILE:
        return task_config

    updated = copy.deepcopy(task_config)
    task_name = updated.get("task_name", "")
    if task_name.startswith("kill_entity:"):
        return _apply_sp_bench_kill(updated)
    if task_name.startswith("mine_block:"):
        return _apply_sp_bench_mine(updated)
    if task_name.startswith("craft_item:"):
        return _apply_sp_bench_craft(updated, smelt_mode=False)
    if task_name.startswith("smelt_item:"):
        return _apply_sp_bench_craft(updated, smelt_mode=True)
    return updated


def summarize_benchmark_profile(task_config: dict, args) -> str:
    profile = getattr(args, "benchmark_profile", "") or ""
    if profile != SP_BENCH_PROFILE:
        return ""

    task_name = task_config.get("task_name", "")
    callback_cfg = task_config.get("callback", {})
    init_inventory_cfg = callback_cfg.get("init_inventory", {})
    init_inventory = init_inventory_cfg.get("init_inventory", [])
    commands = callback_cfg.get("commands", [])
    if task_name.startswith("kill_entity:"):
        mob_spawn_cmds = [cmd for cmd in commands if "summon minecraft:" in cmd]
        return f"commands={len(commands)} summon_cmds={len(mob_spawn_cmds)}"
    if task_name.startswith("mine_block:"):
        tools = [item["type"] for item in init_inventory if _tool_family(item.get("type", ""))]
        return f"tools={tools}"
    if task_name.startswith("craft_item:") or task_name.startswith("smelt_item:"):
        return f"init_inventory_items={len(init_inventory)}"
    return ""


def _apply_sp_bench_kill(task_config: dict) -> dict:
    callback_cfg = task_config.setdefault("callback", {})
    commands = _normalize_commands(callback_cfg.get("commands", []))
    tp_command = commands[0] if commands else ""
    xyz = _extract_tp_xyz(tp_command)
    if xyz is None:
        logging.warning("sp_bench kill profile skipped tp rewrite for %s", task_config.get("task_name"))
        return task_config

    x, y, z = xyz
    entity_name = task_config["task_name"].split(":", 1)[-1]
    revised_commands = [f"/tp @s {x} {y} {z} 180 0"]
    revised_commands.extend(commands[1:])
    revised_commands.append(
        f"/execute as @p at @p run summon minecraft:{entity_name} ^4 ^ ^-4 {{Age:0}}"
    )

    callback_cfg["commands"] = revised_commands
    callback_cfg.pop("mobs", None)
    return task_config


def _apply_sp_bench_mine(task_config: dict) -> dict:
    from openagents.envs.tasks.mine_block import mine_block_spawn_dict

    task_name = task_config.get("task_name", "")
    callback_cfg = task_config.setdefault("callback", {})
    init_inventory_cfg = callback_cfg.setdefault("init_inventory", {})
    init_inventory = list(init_inventory_cfg.get("init_inventory", []))
    if not init_inventory:
        return task_config

    tool_candidates = mine_block_spawn_dict.get(task_name, {}).get("tool", [])
    tool_family = _infer_tool_family(tool_candidates or [init_inventory[0].get("type", "")])
    chosen_tool = _choose_sp_bench_tool(tool_candidates, tool_family)
    if not chosen_tool:
        return task_config

    task_config["callback"]["init_inventory"]["init_inventory"] = [
        {**init_inventory[0], "slot": 0, "type": chosen_tool, "quantity": 1}
    ]
    return task_config


def _apply_sp_bench_craft(task_config: dict, smelt_mode: bool) -> dict:
    callback_cfg = task_config.setdefault("callback", {})
    init_inventory_cfg = callback_cfg.setdefault("init_inventory", {})
    init_inventory = list(init_inventory_cfg.get("init_inventory", []))
    goal_item = _extract_goal_item(task_config)
    distractor_items = _build_distractor_items(init_inventory, goal_item, smelt_mode=smelt_mode)

    occupied_slots = {item["slot"] for item in init_inventory if item.get("slot") != "random"}
    next_slots = _available_slots(occupied_slots, forbidden_slots=init_inventory_cfg.get("forbidden_slots", []))
    for idx, item_type in enumerate(distractor_items):
        slot = next_slots[idx] if idx < len(next_slots) else "random"
        init_inventory.append(
            {
                "slot": slot,
                "type": item_type,
                "quantity": _distractor_quantity(item_type, smelt_mode),
            }
        )

    init_inventory_cfg["init_inventory"] = init_inventory
    return task_config


def _extract_tp_xyz(command: str):
    match = re.match(r"^/tp @s\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", command)
    if not match:
        return None
    values = []
    for token in match.groups():
        values.append(int(float(token)) if float(token).is_integer() else float(token))
    return tuple(values)


def _normalize_commands(commands) -> list[str]:
    normalized = []
    for command in commands or []:
        if isinstance(command, str):
            if command.strip():
                normalized.append(command)
            continue
        if isinstance(command, list):
            for nested in command:
                if isinstance(nested, str) and nested.strip():
                    normalized.append(nested)
    return normalized


def _tool_family(tool_name: str) -> str:
    for family in ("pickaxe", "axe", "shovel", "hoe", "sword"):
        if family in tool_name:
            return family
    return ""


def _infer_tool_family(tool_candidates: list[str]) -> str:
    for tool_name in tool_candidates:
        family = _tool_family(tool_name)
        if family:
            return family
    return ""


def _choose_sp_bench_tool(tool_candidates: list[str], tool_family: str) -> str:
    valid_candidates = [tool for tool in tool_candidates if tool and tool != "air"]
    if not valid_candidates:
        return ""

    if tool_family == "pickaxe":
        return "iron_pickaxe"
    if tool_family == "axe":
        return "iron_axe"
   

    return valid_candidates[0]


def _extract_goal_item(task_config: dict) -> str:
    rewards = task_config.get("rewards", [])
    if rewards and rewards[0].get("objects"):
        return rewards[0]["objects"][0]
    return ""


def _available_slots(occupied_slots: set, forbidden_slots: list[int]) -> list[int]:
    blocked = set(occupied_slots) | set(forbidden_slots or [])
    return [slot for slot in range(1, 36) if slot not in blocked]


def _build_distractor_items(init_inventory: list[dict], goal_item: str, smelt_mode: bool) -> list[str]:
    core_items = {item.get("type") for item in init_inventory if item.get("type")}
    distractors = []
    seed_items = list(core_items)
    if goal_item:
        seed_items.append(goal_item)

    for item_type in seed_items:
        distractors.extend(_related_items(item_type, smelt_mode=smelt_mode))

    fallback_pool = SMELT_DISTRACTOR_ITEMS if smelt_mode else CRAFT_DISTRACTOR_ITEMS
    distractors.extend(fallback_pool)

    unique_items = []
    seen = set(core_items)
    if goal_item:
        seen.add(goal_item)
    for item_type in distractors:
        if not item_type or item_type in seen:
            continue
        seen.add(item_type)
        unique_items.append(item_type)
        if len(unique_items) >= 4:
            break
    return unique_items


def _related_items(item_type: str, smelt_mode: bool) -> list[str]:
    explicit = {
        "coal": ["charcoal", "oak_planks", "stick"],
        "charcoal": ["coal", "oak_planks", "stick"],
        "iron_ingot": ["gold_ingot", "iron_nugget", "gold_nugget"],
        "gold_ingot": ["iron_ingot", "gold_nugget", "iron_nugget"],
        "cobblestone": ["stone", "andesite", "diorite"],
        "redstone": ["redstone_torch", "repeater", "lever"],
        "bow": ["crossbow", "arrow", "string"],
        "stick": ["bamboo", "oak_planks", "ladder"],
        "sand": ["red_sand", "sandstone", "glass"],
        "red_sandstone": ["sandstone", "sand", "red_sand"],
        "quartz_block": ["smooth_quartz", "quartz", "white_concrete"],
        "clay_ball": ["brick", "terracotta", "bricks"],
        "cactus": ["green_dye", "bamboo", "kelp"],
        "chorus_fruit": ["popped_chorus_fruit", "purple_dye", "nether_wart"],
    }
    if item_type in explicit:
        return explicit[item_type]

    if item_type.endswith("_log"):
        wood = item_type[: -len("_log")]
        return [f"{wood}_planks", "oak_log", "spruce_log"]
    if item_type.endswith("_planks"):
        wood = item_type[: -len("_planks")]
        return [f"{wood}_log", "stick", "crafting_table"]
    if item_type.endswith("_ore"):
        return ["coal", "iron_ore", "gold_ore"]
    if item_type.endswith("_wool"):
        return ["white_wool", "string", "carpet"]
    if item_type.endswith("_terracotta"):
        return ["terracotta", "white_terracotta", "clay_ball"]
    if item_type.endswith("_dye"):
        return ["white_dye", "black_dye", "blue_dye"]
    if smelt_mode:
        return ["charcoal", "oak_planks", "sand"]
    return ["stick", "oak_planks", "cobblestone"]


def _distractor_quantity(item_type: str, smelt_mode: bool):
    if item_type in {"coal", "charcoal", "oak_planks", "stick"}:
        return ">=2" if smelt_mode else ">=1"
    if item_type.endswith("_log") or item_type.endswith("_planks"):
        return ">=2"
    return 1
