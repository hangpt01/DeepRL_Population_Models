"""Write hmMDP as a POMDPX XML file."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def write_hmMDPx(TR: List[np.ndarray], REW: np.ndarray, B_FULL: np.ndarray, B_PAR: np.ndarray, GAMMA: float, FILE: str) -> None:
    num_mod = len(TR)
    num_s = REW.shape[0]
    num_a = REW.shape[1]

    ss = [f"state{i}" for i in range(1, num_s + 1)]
    mm = [f"mod{i}" for i in range(1, num_mod + 1)]
    xx = [f"action{i}" for i in range(1, num_a + 1)]

    parts = []
    parts.append('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
    parts.append('<pomdpx version ="1.0" id="sample" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
    parts.append('      xsi:noNamespaceSchemaLocation="pomdpx.xsd">\n')
    parts.append("<Description>hmMDP model</Description>\n")
    parts.append(f"<Discount>{GAMMA}</Discount>\n")
    parts.append("<Variable>\n")
    parts.append('<StateVar vnamePrev="species_0" vnameCurr="species_1" fullyObs="true">\n')
    parts.append(f"<ValueEnum>{' '.join(ss)}</ValueEnum>\n</StateVar>\n")
    parts.append('<StateVar vnamePrev="hidden_0" vnameCurr="hidden_1" fullyObs="false">\n')
    parts.append(f"<ValueEnum>{' '.join(mm)}</ValueEnum>\n</StateVar>\n")
    parts.append('<ActionVar vname="action_control">\n')
    parts.append(f"<ValueEnum>{' '.join(xx)}</ValueEnum>\n</ActionVar>\n")
    parts.append("<ObsVar vname=\"obs\">\n<ValueEnum>o</ValueEnum>\n</ObsVar>\n")
    parts.append('<RewardVar vname="reward_agent" />\n</Variable>\n')

    parts.append("<InitialStateBelief>\n")
    parts.append("<CondProb>\n<Var>species_0</Var>\n<Parent>null</Parent>\n<Parameter type=\"TBL\">\n")
    parts.append("<Entry>\n<Instance> - </Instance>\n")
    parts.append(f"<ProbTable>{' '.join(str(x) for x in B_FULL)}</ProbTable>\n</Entry>\n")
    parts.append("</Parameter>\n</CondProb>\n")
    parts.append("<CondProb>\n<Var>hidden_0</Var>\n<Parent>null</Parent>\n<Parameter type=\"TBL\">\n")
    parts.append("<Entry>\n<Instance> - </Instance>\n")
    parts.append(f"<ProbTable>{' '.join(str(x) for x in B_PAR)}</ProbTable>\n</Entry>\n")
    parts.append("</Parameter>\n</CondProb>\n")
    parts.append("</InitialStateBelief>\n")

    parts.append("<StateTransitionFunction>\n")
    parts.append("<CondProb>\n<Var>species_1</Var>\n<Parent>action_control hidden_0 species_0</Parent>\n<Parameter type=\"TBL\">\n")
    for mod_id in range(num_mod):
        model_text = f"mod{mod_id+1}"
        model_prob_values = TR[mod_id]
        for act_id in range(num_a):
            action_text = f"action{act_id+1}"
            action_prob_values = model_prob_values[:, :, act_id]
            for s0 in range(num_s):
                state0_text = f"state{s0+1}"
                for s1 in range(num_s):
                    state1_text = f"state{s1+1}"
                    parts.append("<Entry>\n<Instance>")
                    parts.append(f"{action_text} {model_text} {state0_text} {state1_text}")
                    parts.append("</Instance>\n")
                    parts.append(f"<ProbTable>{action_prob_values[s0, s1]}</ProbTable>\n</Entry>\n")
    parts.append("</Parameter>\n</CondProb>\n")
    parts.append("<CondProb>\n<Var>hidden_1</Var>\n<Parent>hidden_0</Parent>\n<Parameter type=\"TBL\">\n")
    parts.append("<Entry>\n<Instance> - - </Instance>\n<ProbTable>identity</ProbTable>\n</Entry>\n")
    parts.append("</Parameter>\n</CondProb>\n</StateTransitionFunction>\n")

    parts.append("<ObsFunction>\n<CondProb>\n<Var>obs</Var>\n<Parent>hidden_1</Parent>\n<Parameter type=\"TBL\">\n")
    for mod_id in range(num_mod):
        parts.append("<Entry>\n")
        parts.append(f"<Instance> mod{mod_id+1} o</Instance>\n")
        parts.append("<ProbTable>1</ProbTable>\n</Entry>\n")
    parts.append("</Parameter>\n</CondProb>\n</ObsFunction>\n")

    rew_flat = " ".join(str(x) for x in REW.reshape(-1, order="F"))
    parts.append("<RewardFunction>\n<Func>\n<Var>reward_agent</Var>\n<Parent>action_control species_0</Parent>\n<Parameter type=\"TBL\">\n")
    parts.append("<Entry>\n<Instance> - - </Instance>\n")
    parts.append(f"<ValueTable>{rew_flat}</ValueTable>\n</Entry>\n")
    parts.append("</Parameter>\n</Func>\n</RewardFunction>\n")
    parts.append("</pomdpx>")

    out = Path(FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
