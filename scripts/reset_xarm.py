from xarm.wrapper import XArmAPI

print("Starting: reset_xarm - connecting and preparing to move to go location")

# Replace with your controller IP
arm = XArmAPI('192.168.1.224')

# Connect & clear errors/warnings
arm.connect()
arm.clean_error()
arm.clean_warn()

# Enable motion and set mode/state
arm.motion_enable(enable=True)
arm.set_mode(0)   # position control mode
arm.set_state(0)  # ready

# Move to "home"/initial position (joints)
# Adjust these if your “initial” pose is different
# Example: all zeros in radians
home_joints = [5.3, -80.5, -5.6, 75.3, -22.3, 106.5, -6.5]

print(f"Set to go location (joints): {home_joints}")

# Move slowwwwwly
arm.set_servo_angle(
    angle=home_joints,
    speed=5,       # deg/s
    mvacc=100,      # deg/s^2
    is_radian=False,
    wait=True
)

# Optionally wait until motion finishes
arm.set_state(0)  # ensure back to ready at end
print("Reached go location. Reset complete; disconnecting.")
arm.disconnect()
