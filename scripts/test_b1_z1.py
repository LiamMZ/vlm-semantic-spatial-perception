#!/usr/bin/env python3
"""
B1 + Z1 system test script.

Tests connection and state readout, body height control, and arm primitives.
Does NOT command any walking.

Usage
-----
    # Connection + state only (safe, no motion):
    python scripts/test_b1_z1.py --state-only

    # Full motion test (B1 must be standing on flat ground, arm clear):
    python scripts/test_b1_z1.py

    # Individual components:
    python scripts/test_b1_z1.py --skip-b1
    python scripts/test_b1_z1.py --skip-z1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import os
from dotenv import load_dotenv

# Load .env before anything reads os.environ (including argparse defaults)
load_dotenv(Path(__file__).parents[1] / ".env")

import numpy as np

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.kinematics.b1_robot_interface import B1RobotInterface, Mode
from src.kinematics.z1_robot_interface import Z1RobotInterface

# --------------------------------------------------------------------------- #
# Constants — arm poses
# --------------------------------------------------------------------------- #

# Safe neutral pose: arm roughly upright, gripper centred above the B1 body.
# Adjust after measuring the actual safe pose on your hardware.
Z1_HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Slightly extended forward and down — used to verify joint travel.
Z1_EXTEND = [0.0, 0.5, -0.8, 0.0, 0.5, 0.0]

# Body-height delta range from SDK: -0.13 ~ +0.03 m  (default 0.31 m)
HEIGHT_LOW  = -0.10   # crouch
HEIGHT_HIGH =  0.03   # max raise
HEIGHT_HOLD =  2.0    # seconds to hold each position

STEP_PAUSE  = 1.0     # pause between steps (s)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _sep(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"        {msg}")


def _prompt(msg: str) -> bool:
    """Ask for confirmation. Returns True if user says yes."""
    try:
        ans = input(f"\n  {msg} [y/N]: ").strip().lower()
        return ans in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# --------------------------------------------------------------------------- #
# State tests
# --------------------------------------------------------------------------- #

def test_b1_state(b1: B1RobotInterface) -> bool:
    _sep("B1 — connection & state")
    try:
        state = b1.get_state()
    except Exception as exc:
        _fail(f"get_state() raised: {exc}")
        return False

    _ok("get_state() returned")
    _info(f"mode          : {state.mode}")
    _info(f"battery SOC   : {state.battery_soc} %")
    _info(f"body height   : {state.body_height:.3f} m (delta from default)")
    _info(f"rpy (deg)     : {np.degrees(state.rpy).round(1)}")
    _info(f"position (m)  : {state.position.round(3)}")
    _info(f"foot forces   : {state.foot_force.round(1)}")

    if state.battery_soc < 20:
        print(f"\n  WARNING: battery SOC is low ({state.battery_soc}%)")

    return True


def test_z1_state(z1: Z1RobotInterface) -> bool:
    _sep("Z1 — connection & state")
    try:
        q = z1.get_robot_joint_state()
    except Exception as exc:
        _fail(f"get_robot_joint_state() raised: {exc}")
        return False

    if q is None:
        _fail("getQ returned None — check Z1 is powered and reachable")
        return False

    _ok("getQ returned joint positions")
    _info(f"joints (rad)  : {np.round(q, 3)}")
    _info(f"joints (deg)  : {np.degrees(q).round(1)}")

    state = z1.get_robot_state()
    _info(f"state stamp   : {state['stamp']:.3f}")
    return True


# --------------------------------------------------------------------------- #
# Motion tests
# --------------------------------------------------------------------------- #

def test_b1_height(b1: B1RobotInterface) -> bool:
    _sep("B1 — body height control (no walking)")

    print(f"\n  Will command: crouch ({HEIGHT_LOW:+.2f} m) then raise ({HEIGHT_HIGH:+.2f} m).")
    print("  The robot will NOT walk. Keep the area around legs clear.")
    if not _prompt("Proceed with height test?"):
        print("  Skipped.")
        return True

    try:
        # Make sure we're in a safe starting state
        print("\n  Entering force-stand at default height...")
        b1.set_body_pose(body_height=0.0, duration=1.0)
        time.sleep(STEP_PAUSE)

        print(f"  Crouching to {HEIGHT_LOW:+.2f} m delta...")
        b1.set_body_pose(body_height=HEIGHT_LOW, duration=HEIGHT_HOLD)
        _ok(f"Held crouch for {HEIGHT_HOLD:.1f}s")
        time.sleep(STEP_PAUSE)

        print(f"  Raising to {HEIGHT_HIGH:+.2f} m delta...")
        b1.set_body_pose(body_height=HEIGHT_HIGH, duration=HEIGHT_HOLD)
        _ok(f"Held raised for {HEIGHT_HOLD:.1f}s")
        time.sleep(STEP_PAUSE)

        print("  Returning to default height...")
        b1.set_body_pose(body_height=0.0, duration=1.0)
        _ok("Height test complete")
        return True

    except KeyboardInterrupt:
        print("\n  Interrupted — sending idle to stop")
        b1.stop(hold=0.5)
        return False
    except Exception as exc:
        _fail(f"Height test raised: {exc}")
        b1.stop(hold=0.5)
        return False


def test_z1_arm(z1: Z1RobotInterface) -> bool:
    _sep("Z1 — arm control")

    print(f"\n  Will move arm: home → extend → home.")
    print(f"  Home   : {Z1_HOME}")
    print(f"  Extend : {Z1_EXTEND}")
    print("  Ensure the arm workspace is clear of obstacles.")
    if not _prompt("Proceed with arm test?"):
        print("  Skipped.")
        return True

    try:
        print("\n  Moving to home pose...")
        resp = z1.move_j(Z1_HOME, gripper_pos=0, speed=0.3)
        if resp is None:
            _fail("move_j(home) got no response")
            return False
        _ok(f"move_j(home) → HTTP {resp.status_code}")
        time.sleep(2.0)

        print("  Opening gripper...")
        resp = z1.open_gripper(speed=128, force=128)
        _ok(f"open_gripper → HTTP {resp.status_code if resp else 'no response'}")
        time.sleep(1.0)

        print("  Moving to extended pose...")
        resp = z1.move_j(Z1_EXTEND, gripper_pos=0, speed=0.3)
        if resp is None:
            _fail("move_j(extend) got no response")
            return False
        _ok(f"move_j(extend) → HTTP {resp.status_code}")
        time.sleep(2.0)

        print("  Closing gripper...")
        resp = z1.close_gripper(speed=128, force=128)
        _ok(f"close_gripper → HTTP {resp.status_code if resp else 'no response'}")
        time.sleep(1.0)

        print("  Returning to home...")
        resp = z1.move_j(Z1_HOME, gripper_pos=0, speed=0.3)
        _ok(f"move_j(home) → HTTP {resp.status_code if resp else 'no response'}")
        time.sleep(2.0)

        print("  Opening gripper (rest position)...")
        z1.open_gripper(speed=128, force=64)
        time.sleep(0.5)

        _ok("Arm test complete")
        return True

    except KeyboardInterrupt:
        print("\n  Interrupted — commanding home")
        z1.move_j(Z1_HOME, speed=0.2)
        return False
    except Exception as exc:
        _fail(f"Arm test raised: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B1 + Z1 system test script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--state-only",
        action="store_true",
        help="Only test connections and read state — no motion commands",
    )
    p.add_argument(
        "--skip-b1",
        action="store_true",
        help="Skip all B1 tests (useful when only Z1 is connected)",
    )
    p.add_argument(
        "--skip-z1",
        action="store_true",
        help="Skip all Z1 tests",
    )
    p.add_argument(
        "--b1-ip",
        default="192.168.12.159",
        help="B1 high-level SDK server IP (default: 192.168.123.161)",
    )
    p.add_argument(
        "--z1-url",
        default="http://192.168.12.159:12000/unitree/z1",
        help="Z1 HTTP API URL",
    )
    p.add_argument(
        "--sdk-path",
        default=os.environ.get("UNITREE_SDK_PATH"),
        help="Path to unitree_legged_sdk Python bindings directory (default: $UNITREE_SDK_PATH)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results: dict[str, bool] = {}

    print("\n========================================")
    print("  B1 + Z1 System Test")
    print("========================================")
    if args.state_only:
        print("  Mode: STATE ONLY (no motion)")
    else:
        print("  Mode: FULL (state + motion)")
    print(f"  B1 IP  : {args.b1_ip}")
    print(f"  Z1 URL : {args.z1_url}")

    # ------------------------------------------------------------------ #
    # B1
    # ------------------------------------------------------------------ #
    b1: B1RobotInterface | None = None
    if not args.skip_b1:
        _sep("B1 — initialising")
        try:
            b1 = B1RobotInterface(server_ip=args.b1_ip, sdk_path=args.sdk_path)
            _ok("B1RobotInterface created")
        except ImportError as exc:
            _fail(str(exc))
            print("  Tip: set --sdk-path to the directory containing robot_interface.so")
            results["b1_init"] = False
        except Exception as exc:
            _fail(f"Unexpected error: {exc}")
            results["b1_init"] = False

        if b1 is not None:
            results["b1_init"] = True
            results["b1_state"] = test_b1_state(b1)

            if not args.state_only and results.get("b1_state"):
                results["b1_height"] = test_b1_height(b1)
    else:
        print("\n  [--skip-b1] Skipping all B1 tests.")

    # ------------------------------------------------------------------ #
    # Z1
    # ------------------------------------------------------------------ #
    z1: Z1RobotInterface | None = None
    if not args.skip_z1:
        _sep("Z1 — initialising")
        try:
            z1 = Z1RobotInterface(url=args.z1_url)
            _ok("Z1RobotInterface created")
            results["z1_init"] = True
        except Exception as exc:
            _fail(f"Unexpected error: {exc}")
            results["z1_init"] = False

        if z1 is not None:
            results["z1_state"] = test_z1_state(z1)

            if not args.state_only and results.get("z1_state"):
                results["z1_arm"] = test_z1_arm(z1)
    else:
        print("\n  [--skip-z1] Skipping all Z1 tests.")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    _sep("Results")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        mark = "✓" if passed else "✗"
        print(f"  {mark}  {name:<20} {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All tests passed.")
    else:
        print("  Some tests failed — see output above.")

    # Cleanup
    if b1 is not None:
        try:
            b1.stop(hold=0.3)
        except Exception:
            pass

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
