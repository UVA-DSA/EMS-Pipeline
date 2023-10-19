
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


ems_interventions = {"medical - chest pain - cardiac suspected (protocol 2 - 1)":adult_cardiact_arrest_protocol,
                             'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress)' :respiratory_distress_protocol}

