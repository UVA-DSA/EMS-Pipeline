# information about this lib: https://sites.google.com/site/pydatalog/
from pyDatalog import pyDatalog

# create terms
pyDatalog.create_terms('RH, PH, C, BR, P, BL, op, Y')

# rules
(op[RH,PH,C,BR,P,BL] == 'Eliminate_Harzard') <= ('Rescuer_Hazard' == RH)
(op[RH,PH,C,BR,P,BL] == 'Eliminate_Harzard') <= ('Rescuer_Hazard' != RH) & ('Patient_Hazard' == PH)
(op[RH,PH,C,BR,P,BL] == 'Secondary_Survey') <= ('Rescuer_Hazard' != RH) & ('Patient_Hazard' != PH) & \
                                            ('Conscious' == C) & ('Bleeding' != BL)
(op[RH,PH,C,BR,P,BL] == 'Control_Bleeding') <= ('Rescuer_Hazard' != RH) & ('Patient_Hazard' != PH) & \
                                            ('Conscious' == C) & ('Bleeding' == BL)
(op[RH,PH,C,BR,P,BL] == 'Open_the_Airway, Secondary_Survey') <= ('Rescuer_Hazard' != RH) & \
                                            ('Patient_Hazard' != PH) & ('Conscious' != C) & ('Breathing' == BR) & ('Pulse' == P) & ('Bleeding' != BL)
(op[RH,PH,C,BR,P,BL] == 'Open_the_Airway, Control_Bleeding') <= ('Rescuer_Hazard' != RH) & \
                                            ('Patient_Hazard' != PH) & ('Conscious' != C) & ('Breathing' == BR) & ('Pulse' == P) & ('Bleeding' == BL)
(op[RH,PH,C,BR,P,BL] == 'Open_the_Airway, Start_Artificial_Ventilation, \
                                            Secondary_Survey') <= ('Rescuer_Hazard' != RH) & \
                                            ('Patient_Hazard' != PH) & ('Conscious' != C) & ('Breathing' != BR) & ('Pulse' == P) & ('Bleeding' != BL)
(op[RH,PH,C,BR,P,BL] == 'Open_the_Airway, Start_Artificial_Ventilation, \
                                            Control_Bleeding') <= ('Rescuer_Hazard' != RH) & \
                                            ('Patient_Hazard' != PH) & ('Conscious' != C) & ('Breathing' != BR) & ('Pulse' == P) & ('Bleeding' == BL)
(op[RH,PH,C,BR,P,BL] == 'Open_the_Airway, Start_External_Cardiac_Compressions, Secondary_Survey') \
                                            <= ('Rescuer_Hazard' != RH) & \
                                            ('Patient_Hazard' != PH) & ('Conscious' != C) & ('Breathing' == BR) & ('Pulse' != P) & ('Bleeding' != BL)
(op[RH,PH,C,BR,P,BL] == 'Open_the_Airway, Start_External_Cardiac_Compressions, Control_Bleeding') \
                                            <= ('Rescuer_Hazard' != RH) & \
                                            ('Patient_Hazard' != PH) & ('Conscious' != C) & ('Breathing' == BR) & ('Pulse' != P) & ('Bleeding' == BL)
(op[RH,PH,C,BR,P,BL] == 'Open_the_Airway, Start_Artificial_Ventilation, Start_External_Cardiac_Compressions,\
                                            Secondary_Survey') <= ('Rescuer_Hazard' != RH) & \
                                            ('Patient_Hazard' != PH) & ('Conscious' != C) & ('Breathing' != BR) & ('Pulse' != P) & ('Bleeding' != BL)
(op[RH,PH,C,BR,P,BL] == 'Open_the_Airway, Start_Artificial_Ventilation, Start_External_Cardiac_Compressions,\
                                            Control_Bleeding') <= ('Rescuer_Hazard' != RH) & \
                                            ('Patient_Hazard' != PH) & ('Conscious' != C) & ('Breathing' != BR) & ('Pulse' != P) & ('Bleeding' == BL)
											
# implement logic function
result = op['Rescuer_Hazard','','','uBreathing','uPulse','uBleeding'] == Y

# show response
print result[0][0]