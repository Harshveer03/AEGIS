import pytest
import uuid
from backend.brain import brain, SessionContext
from backend.registry import agent_registry, skill_registry
from backend.models.task import TaskOutcome
from backend.memory.task_store import task_store
from backend.skills.executor import skill_executor


def make_session():
    return SessionContext(session_id=str(uuid.uuid4()))


# ── Acceptance Test 1 ────────────────────────────────────────────
# A task routed to an existing agent executes correctly end to end
# and the task object persists to the task store.

def test_1_task_routes_and_persists():
    session = make_session()
    result = brain.process("list the current directory", session)

    assert result["status"] == "completed", f"Expected completed, got {result['status']}"
    assert result["agent"] == "file_agent", f"Expected file_agent, got {result['agent']}"
    assert result["outcome"] == TaskOutcome.SUCCESS

    # Verify task was persisted
    task = task_store.get(result["task_id"])
    assert task is not None, "Task not found in task store"
    assert task.task_id == result["task_id"]
    assert "file_agent" in task.agents_used

    print(f"  ✓ Task {result['task_id'][:8]} routed to file_agent, persisted to store")


# ── Acceptance Test 2 ────────────────────────────────────────────
# A novel task triggers the Agent Factory, a new agent is created
# and registered, and the registry reflects the new agent.

def test_2_agent_factory_creates_agent():
    from backend.factory.agent_factory import get_agent_factory
    import time

    # Use a unique capability each run to avoid collision with previous runs
    unique_capability = f"fetch and summarise weather data from a weather api {int(time.time())}"

    factory = get_agent_factory(brain._llm_call)
    success, message, spec = factory.create(unique_capability)

    assert success, f"Agent Factory failed: {message}"
    assert spec is not None
    assert spec.name.endswith("_agent"), f"Agent name should end in _agent, got {spec.name}"
    assert len(spec.skills) > 0, "Agent should have at least one skill"
    assert agent_registry.exists(spec.name), "Agent not found in registry after creation"
    assert spec.is_generated, "Agent should be marked as generated"

    # Verify it can be retrieved from registry
    retrieved = agent_registry.get(spec.name)
    assert retrieved is not None
    assert retrieved.name == spec.name

    print(f"  ✓ Agent '{spec.name}' created with skills: {spec.skills}")


# ── Acceptance Test 3 ────────────────────────────────────────────
# A novel skill requirement triggers the Skill Factory, the skill
# is created, validated, and registered in the skill registry.

def test_3_skill_factory_creates_skill():
    from backend.factory.skill_factory import get_skill_factory
    import time

    # Use unique capability each run to avoid collision with previous runs
    unique_capability = f"multiply two numbers together and return the product {int(time.time())}"

    factory = get_skill_factory(brain._llm_call)
    success, message, spec = factory.create(unique_capability)

    assert success, f"Skill Factory failed: {message}"
    assert spec is not None
    assert skill_registry.exists(spec.name), "Skill not found in registry after creation"
    assert spec.is_generated, "Skill should be marked as generated"

    # Verify the skill is actually callable
    fn = skill_registry.resolve_callable(spec.name)
    assert callable(fn), "Registered skill is not callable"

    print(f"  ✓ Skill '{spec.name}' created, registered, and callable")


# ── Acceptance Test 4 ────────────────────────────────────────────
# A write operation hits the confirmation gate and does not execute
# until confirmed. The gate blocks correctly.

def test_4_confirmation_gate_blocks_write():
    # Attempt write without confirmation — should be blocked
    result = skill_executor.execute(
        "write_file",
        {"path": "test_gate.txt", "content": "should not be written"},
        task_id="gate_test",
        auto_confirm=False,
    )

    assert not result.success, "Write should have been blocked"
    assert result.error == "AWAITING_CONFIRMATION", f"Expected AWAITING_CONFIRMATION, got {result.error}"
    assert result.data.get("requires_confirmation") is True

    # Verify file was NOT written
    from pathlib import Path
    assert not Path("test_gate.txt").exists(), "File was written without confirmation — gate failed"

    # Now confirm and execute — should succeed
    confirmed = skill_executor.confirm_and_execute(
        "write_file",
        {"path": "test_gate_confirmed.txt", "content": "confirmed write"},
        task_id="gate_test",
    )

    assert confirmed.success, f"Confirmed write failed: {confirmed.error}"
    assert Path("test_gate_confirmed.txt").exists(), "File not written after confirmation"

    # Cleanup
    Path("test_gate_confirmed.txt").unlink(missing_ok=True)

    print("  ✓ Confirmation gate blocked write, confirmed write succeeded")


# ── Runner ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("AEGIS Phase 1 Acceptance Tests")
    print("="*60)

    tests = [
        ("Test 1: Task routes and persists",        test_1_task_routes_and_persists),
        ("Test 2: Agent Factory creates agent",     test_2_agent_factory_creates_agent),
        ("Test 3: Skill Factory creates skill",     test_3_skill_factory_creates_skill),
        ("Test 4: Confirmation gate blocks write",  test_4_confirmation_gate_blocks_write),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{name}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n✓ PHASE 1 COMPLETE — All acceptance tests passed.")
    else:
        print(f"\n✗ {failed} test(s) need attention.")