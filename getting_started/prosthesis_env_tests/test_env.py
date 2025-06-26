import os
import jax
import time
import mujoco

# from loco_mujoco.task_factories import ImitationFactory


# # can increase the speed by ~30% on some GPUs
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True ')


# # create env
# env = ImitationFactory.make("MjxSkeletonMuscle", default_dataset_conf=dict(task="walk"))

# # get actuator_dyntype
# actuator_dyntype = env.get_actuator_dyntype()
# print(f"Actuator Dyntype: {actuator_dyntype}")





model = mujoco.MjModel.from_xml_path('/home/nadinebadie/loco-mujoco/loco_mujoco/models/skeleton/skeleton_muscle.xml') #'/home/nadinebadie/loco-mujoco/loco_mujoco/models/myo_model/myoskeleton/myoskeleton.xml') #'/home/nadinebadie/loco-mujoco/loco_mujoco/models/skeleton/skeleton_muscle.xml')

spec = mujoco.MjSpec.from_file('/home/nadinebadie/loco-mujoco/loco_mujoco/models/skeleton/skeleton_muscle.xml') #'/home/nadinebadie/loco-mujoco/loco_mujoco/models/myo_model/myoskeleton/myoskeleton.xml') #'/home/nadinebadie/loco-mujoco/loco_mujoco/models/skeleton/skeleton_muscle.xml')

# # 
calcn = spec.find_body('calcn_r')

print(f"Calcn Body: {calcn}")

# get next body after calcn
next_body = calcn.next_body
print(f"Next Body Name: {dir(next_body)}")


# get attributes of the calcn body
# print(f"attributes: {dir(calcn)}")
# print(f"calcn site: {calcn.sites}")

# names_site_final =[]
# for site in calcn.sites:
#     names_site = site.name
#     # print(f"Site Name: {names_site}")
#     if '-P' in names_site:
#         names_site_final.append(names_site[:-3])
#         # print(f"Site: {names_site_final}")
#     # get all tendons associated with the site
#     # print(f"site attributes: {dir(site)}")

# print(f"Site Names: {names_site_final}")
# print(len(names_site_final))

# # take out repeated sites
# names_site_final = set(names_site_final)
# # print(f"Final Site Names: {names_site_final}")
# print(len(names_site_final))
# print(f"Final Site Names: {list(names_site_final)}")



# for m in spec.actuators:
# #     print(f"Muscle Name: {m.name}")
#     if m.name in names_site_final:
#         print(f"Muscle associated with site {m.name}")
# print(list(names_site_final)[0])
# for t in spec.tendons:
#     # print(f"Tendon Name: {t.name}")
#     for i in range(len(names_site_final)):
#         if list(names_site_final)[i] in t.name:
#             print(f"Tendon associated with site {list(names_site_final)[i]}: {t.name}")
   
   
   
   
   
   
   
   
   
   
   
    #     # get all muscles associated with the tendon
    #     # print(f"Tendon Muscles: {[m.name for m in t.muscles]}")
    # # if t.name in names_site_final:
    # #     print(f"Tendon associated with site {names_site_final}: {t.name}")
    #     # get all tendons associated with the muscle
        # print(f"Muscle Tendons: {[tendon.name for tendon in m.tendons]}")

# # get all tendons 
# tendons = spec.tendons
# # tendons[0].wrap_site("test")
# # Attempt to retrieve site information by calling wrap_site
# for tendon in tendons:
#     if tendon.name == "glut_med1_r_tendon":
#         print(f"Tendon Name: {tendon.name}")
#         if tendon.wrap_site is not None:
#             if callable(tendon.wrap_site):
#                 try:
#                     # Call wrap_site with the correct argument (e.g., tendon name or other relevant input)
#                     site_info = tendon.wrap_site(tendon.name)
#                     print(f"Wrap Site Info: {site_info}")
#                     # Inspect the attributes of the returned site_info
#                     print(f"Attributes of site_info: {dir(site_info)}")
#                 except TypeError as e:
#                     print(f"Error calling wrap_site: {e}")
#             else:
#                 print("Wrap Site is not callable.")
#         else:
#             print("No associated wrap site.")
# # tendon = spec.find_tendon('tibialis_anterior_r')
# # print(f"Tendon: {tendon}")
# # print(f"Tendon attributes: {dir(tendons)}")
# # print(f"Tendons: {[dir(tendon) for tendon in tendons]}")


# # mjtWrap_test = mujoco.mjtWrap.mjWRAP_SITE  # Example of how to use mjtWrap
# # print(f"mjtWrap_test: {mjtWrap_test}")



# # Test to get full kinematic chain starting from talus 
# for j in spec.joints:
#     # print(f"Joint Name: {j.name}")
#     if j.name in 'ankle_angle_r':
#         ankle_angle_r = j
#         print(f"Talus Body: {ankle_angle_r}")
#         # # get attributes of the talus body
#         print(f"attributes: {dir(ankle_angle_r)}")

# spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
# print(f"Worldbody Geoms: {[geom.name for geom in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)]}")






# # Actuator Dyntype
# # model_dyntype = model.actuator_dyntype
# # print(f"Model Actuator Dyntype: {model_dyntype}")

# # for i in range(model.nu):
# #     if model_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
# #         print(i, 'Muscle')
# #     else: 
# #         print(i, 'Motor')



# # all site_ids associated with joint ankle 
# # ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'ankle')
# # subtal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'subtalar')
# # mtp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'mtp')
# # print(f"Ankle Site IDs: {ankle_id}")
# # print(f"Subtalar Site IDs: {subtal_id}")
# # print(f"MTP Site IDs: {mtp_id}")

# # print([model.body(i).name for i in range(model.nbody)])
