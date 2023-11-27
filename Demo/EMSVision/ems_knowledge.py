
from EMSVision.ems_action_keywords import *




adult_cardiact_arrest_protocol = {"cv_ecg_12_lead":cv_ecg_12_lead,
                                  "cv_defibrillation_manual":cv_defibrillation_manual,
                                  "cpr_manual":cpr_manual,
                                  "iv_access":iv_access,
                                  "adv_airway_capnography":adv_airway_capnography,
                                  "admin_epi":admin_epi
                                  }


respiratory_distress_protocol = {"resp_assist_ventilation_bvm_via_mask":resp_assist_ventilation_bvm_via_mask,
                                 "resp_nebulizer_therapy":resp_nebulizer_therapy,
                                 "resp_airway_adjunct":resp_airway_adjunct,
                                 "resp_endotracheal_tube":resp_endotracheal_tube,
                                 "resp_administer_albuterol":resp_administer_albuterol,
                                }

tachycardia_protocol = {"resp_assist_ventilation_bvm_via_mask":resp_assist_ventilation_bvm_via_mask,
                                 "resp_airway_adjunct":resp_airway_adjunct,
                                 "resp_endotracheal_tube":resp_endotracheal_tube,
                                 "cv_ecg_12_lead":cv_ecg_12_lead,
                                 "cardiac_monitor":cardiac_monitor,
                        }

ems_interventions = {"medical - chest pain - cardiac suspected (protocol 2 - 1)":adult_cardiact_arrest_protocol,
                    "general - cardiac arrest (protocol 2 - 7)":adult_cardiact_arrest_protocol,
                    "general - cardiac arrest (protocol 8 - 2)":adult_cardiact_arrest_protocol,
                    "medical - supraventricular tachycardia (including atrial fibrillation) medical - tachycardia medical - ventricular tachycardia with a pulse (protocol 2 - 8)":tachycardia_protocol,
                    'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress) (protocol 3 - 11)':respiratory_distress_protocol}

