import xml.etree.ElementTree as ET
import re


# Convert the numbers to float and create a list
key_string = "<key qpos='0 0 0 0 0 -1.63363 .8526 -1.6 -1.6419 -.3 -0.09426 2.48218 0.6495 -0.01938 0 0 1.57 1.5936 .4 0 0 0.9 0 0 0 0 0.59089 0.7 1 0 0 0 0 0 -0.5235 0.6 -0.8 0 0.2 0 0 0'/>"
numbers = re.findall(r"[-+]?\d*\.\d+|\d+", key_string)
refs = [float(num) for num in numbers]

# Parse the XML file
tree = ET.parse('stompy_og.xml')
root = tree.getroot()

# Find all 'joint' elements
joints = root.findall('.//joint')

# Update the 'ref' attribute with the new values
for joint, new_ref in zip(joints, refs):
    joint.set('ref', str(new_ref))

# Save the modified XML back to a file
tree.write('stompy.xml')