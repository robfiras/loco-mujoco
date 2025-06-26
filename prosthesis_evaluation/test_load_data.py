import pickle 
import matplotlib.pyplot as plt

with open("/home/nadinebadie/loco-mujoco/prosthesis_test/outputs/2025-06-11/16-30-59/20250625_140700_evaluation_results_1000steps.pkl", "rb") as f:
    loaded_data = pickle.load(f)

all_sensor_data = loaded_data.get("all_sensor_force")
left_knee_torque_sensor = all_sensor_data["left_knee_mimic_torque_sensor"]
print("left_knee_torque_sensor", left_knee_torque_sensor[0:10])
right_knee_torque_sensor = all_sensor_data["right_knee_mimic_torque_sensor"]
right_foot_torque_sensor = all_sensor_data["right_foot_mimic_torque_sensor"]
left_foot_torque_sensor = all_sensor_data["left_foot_mimic_torque_sensor"]

plt.figure(figsize=(12, 6))
plt.plot(left_knee_torque_sensor, label='Left Knee Torque Sensor', alpha=0.5)
plt.plot(right_knee_torque_sensor, label='Right Knee Torque Sensor', alpha=0.5)
plt.plot(left_foot_torque_sensor, label='Left Foot Torque Sensor', alpha=0.5)
plt.plot(right_foot_torque_sensor, label='Right Foot Torque Sensor', alpha=0.5)
plt.xlabel('Time Step')
plt.ylabel('Torque Sensor Values')
plt.title('Torque Sensor Values Over Time')
plt.legend()
plt.grid()
plt.savefig('torque_sensor_plot.png')

