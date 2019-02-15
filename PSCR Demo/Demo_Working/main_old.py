import py_trees
import behaviours as be
from py_trees.blackboard import Blackboard

def NLP():
    behaviour_tree.tick_tock(
            sleep_ms=50,
            number_of_iterations=1,
            pre_tick_handler=None,
       post_tick_handler=None
        )

if __name__ == '__main__':
    # build the tree
    root = py_trees.composites.Sequence("Root")
    selector = py_trees.composites.Selector("ProtocolSelector")
    BLSCPR_C = be.BLSCPR_Condition()
    ChestPain_C = be.ChestPain_Condition()
    BC_C = be.Bleeding_Control_Condition()
    ALS_C = be.AlteredMentalStatus_Condition()
    BI_C = be.Burn_Injury_Condition()
    GTG_C = be.GeneralTraumaGuideline_Condition()
    RD_C = be.RespiratoryDistress_Condition()
    IG = be.InformationGathering()
    TC = be.TextCollection()
    root.add_children([TC,IG,selector])
    selector.add_children([BLSCPR_C,ALS_C,BC_C,BI_C,GTG_C,RD_C,ChestPain_C])
    behaviour_tree = py_trees.trees.BehaviourTree(root)
    behaviour_tree.setup(15)
    NLP()
    