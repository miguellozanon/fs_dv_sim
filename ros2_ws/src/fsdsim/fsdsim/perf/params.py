"""LR26 Performance parameters with the LapSim (MATLAB) field names.

One parameter set drives both the quasi-steady-state lap time model (`fsdsim.perf.qss`,
a port of the Performance department's LapSim) and the dynamic driverless simulator
(`fsdsim.dynamics` in `performance` mode), so a parameter change shows up in both.

    p = PerfParams.from_yaml("performance_lr26.yaml")
    p2 = p.with_changes("m=300", "mu*1.05")       # absolute value, or scale with *
    PerfParams.from_matlab("/mnt/c/.../LR26_LapSim/config/lr26_params.m")   # import the original
"""

from __future__ import annotations

import ast
import dataclasses
import math
import operator
import re
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class PerfParams:
    # Defaults = Performance documentation v2.0, Table 3.1 (= LR26_LapSim/config/lr26_params.m).
    # --- mass and geometry ---------------------------------------------------
    m: float = 284.0            # [kg] car + driver (LR26 goal)
    g: float = 9.81
    lf: float = 0.85065         # [m] CoG -> front axle
    lr: float = 0.75435         # [m] CoG -> rear axle
    h_cog: float = 0.265        # [m]
    sf: float = 1.2264          # [m] front track width
    sr: float = 1.2264          # [m] rear track width
    e_i: float = 1.0            # rotating-mass factor (m_eff = m * e_i) for longitudinal motion
    # --- tyres: F_pot = Fz * (mu + dmu/dFz * (Fz - Fz0)) * util -----------------
    mux: float = 1.43
    muy: float = 1.64
    dmux_dFz: float = -2.28e-5  # [1/N]
    dmuy_dFz: float = -1.47e-4  # [1/N]
    Fz0: float = 700.0          # [N] reference load of mu
    tire_exp: float = 2.0       # friction-ellipse exponent
    util_x: float = 1.0         # driver friction utilisation (1 = tyre limit; events override it)
    util_y: float = 1.0
    # --- aero (the efficiency factor scales drag AND downforce) ---------------
    rho: float = 1.18
    A: float = 1.044            # [m^2] frontal area
    c_w: float = 1.38           # drag coefficient
    c_z: float = 4.15           # downforce coefficient
    aero_eff: float = 0.80
    aero_bal_f: float = 0.485   # front share of downforce
    f_roll: float = 0.005       # rolling resistance coefficient
    # --- powertrain (per driven axle = one motor set) --------------------------
    circ_ref: float = 1.57      # [m] loaded tyre circumference (r = circ_ref / 2 pi)
    i_trans: float = 14.5       # gear ratio
    T_motor: float = 38.0       # [Nm] peak motor torque per driven axle
    n_max: float = 21000.0      # [rpm] motor speed limit
    P_max: float = 80000.0      # [W]
    P_at_accumulator: float = 0.0   # 1: P_max is accumulator power (x eta_motor at the motor)
    eta_g: float = 0.95         # gearbox efficiency
    eta_motor: float = 0.90
    drive_awd: float = 0.0      # 0 = RWD, 1 = identical motor set on the front axle
    # --- brakes / energy -------------------------------------------------------
    brake_bias: float = math.nan    # front share; NaN = ideal balance
    P_regen: float = 0.0        # [W] recuperation limit (off, as in the documentation)
    eta_regen: float = 0.80
    # --- LapSim event / strategy settings (not used by the dynamic car) --------
    P_max_endurance: float = 18000.0        # [W] power cap in endurance
    endurance_torque_scale: float = 1.0     # T_motor multiplier in endurance
    v_lim: float = math.inf                 # [m/s] speed limit
    ax_max_launch: float = 7.0              # [m/s^2] launch estimate (lts_launch_check)
    # --- bookkeeping -----------------------------------------------------------
    source: str = "provisional"  # where the values come from (see performance_lr26.yaml)
    events: dict = field(default_factory=dict)   # per-event overrides, e.g. {"skidpad": {"util_y": 0.98}}

    # ------------------------------------------------------------------ #
    @property
    def l(self) -> float:  # noqa: E743
        return self.lf + self.lr

    @property
    def r(self) -> float:
        return self.circ_ref / (2 * math.pi)

    def numeric_fields(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self) if f.type in ("float", float)]

    def copy(self, **changes) -> "PerfParams":
        return dataclasses.replace(self, **changes)

    def for_event(self, event: str) -> "PerfParams":
        """Parameters with the per-event overrides applied (e.g. driver utilisation)."""
        ov = (self.events or {}).get(event) or {} if event else {}
        return self.copy(**{k: float(v) for k, v in ov.items()}) if ov else self

    # ------------------------------------------------------------------ #
    def with_changes(self, *changes: str) -> "PerfParams":
        """Apply "name=value", "name*factor", "name+delta" or "name-delta" changes (LapSim names or aliases)."""
        p = self
        for ch in changes:
            for part in [c for c in re.split(r"[,\s]+", ch) if c]:
                p = p._apply(part)
        return p

    def _apply(self, change: str) -> "PerfParams":
        m = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*([=*+-])\s*(.+?)\s*", change)
        if not m:
            raise ValueError(f"cannot parse parameter change '{change}' (use name=value, name*factor, name+delta, name-delta)")
        name, op, raw = m.groups()
        val = _eval(raw, {})
        targets = ALIASES.get(name.lower(), (name,))
        if targets == ("mux", "muy") and op == "=":
            # as lts_set_param(veh, 'mu', value): muy = value, mux keeps its ratio to muy
            return self.copy(muy=val, mux=val * self.mux / self.muy)
        changes = {}
        for t in targets:
            if t not in {f.name for f in dataclasses.fields(self)}:
                raise KeyError(f"unknown parameter '{name}'. Known: {', '.join(sorted(self.numeric_fields()))}; "
                               f"aliases: {', '.join(sorted(ALIASES))}")
            cur = float(getattr(self, t))
            changes[t] = val if op == "=" else cur * val if op == "*" else cur + val if op == "+" else cur - val
        return self.copy(**changes)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: (None if isinstance(v, float) and math.isnan(v) else
                    ("inf" if v > 0 else "-inf") if isinstance(v, float) and math.isinf(v) else v)
                for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict | None) -> "PerfParams":
        d = dict(d or {})
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            import warnings

            warnings.warn(f"performance parameters: ignoring unknown keys {unknown}", stacklevel=2)
        kw = {}
        for k in names & set(d):
            v = d[k]
            if k in ("source",):
                kw[k] = str(v)
            elif k == "events":
                kw[k] = dict(v or {})
            else:
                kw[k] = math.nan if v is None or (isinstance(v, str) and v.lower() == "nan") else float(v)
        return cls(**kw)

    @classmethod
    def from_yaml(cls, path: str) -> "PerfParams":
        if yaml is None:
            raise RuntimeError("pyyaml is needed to read " + path)
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_matlab(cls, path: str, base: "PerfParams | None" = None) -> tuple["PerfParams", dict]:
        """Read `veh.<field> = <expr>;` assignments from lr26_params.m.

        Returns (params, report) where report lists imported, unknown and unparsed fields.
        Expressions may use numbers, + - * / ^, parentheses, pi, NaN, Inf, true/false and
        fields assigned earlier (veh.lf + veh.lr ...)."""
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        base = base or cls()
        names = {f.name for f in dataclasses.fields(cls)}
        env: dict[str, float] = {}
        imported, unknown, failed = {}, {}, {}
        for code in matlab_statements(text):
            mm = re.match(r"^(\w+)\.(\w+)\s*=\s*(.+?)\s*$", code)
            if not mm:
                continue
            _struct, fld, expr = mm.groups()
            try:
                val = _eval(expr, env)
            except Exception as e:  # noqa: BLE001
                failed[fld] = f"{expr}  ({e})"
                continue
            env[fld] = val
            if fld in names and fld not in ("source", "events"):
                imported[fld] = val
            else:
                unknown[fld] = val
        p = base.copy(**imported, source=f"imported from {path}")
        return p, {"imported": imported, "unknown": unknown, "failed": failed}


# Same aliases as LR26_LapSim's lts_set_param (plus a few long names), case-insensitive.
ALIASES = {
    "mu": ("mux", "muy"),
    "mass": ("m",),
    "gear": ("i_trans",), "gear_ratio": ("i_trans",),
    "cl": ("c_z",), "cz": ("c_z",), "downforce": ("c_z",),
    "cd": ("c_w",), "cw": ("c_w",), "drag": ("c_w",),
    "cog": ("h_cog",), "cog_height": ("h_cog",),
    "power": ("P_max",),
    "cop": ("aero_bal_f",), "aero_balance": ("aero_bal_f",),
    "torque": ("T_motor",),
}


def _strip_comment(line: str) -> str:
    """MATLAB line without its % comment (a % inside a 'string' is kept)."""
    out, in_str = [], False
    for i, ch in enumerate(line):
        if ch == "'":
            prev = line[i - 1] if i else " "
            if in_str or not (prev.isalnum() or prev in ")]}_.'"):   # ' after a value is a transpose
                in_str = not in_str
        elif ch == "%" and not in_str:
            break
        out.append(ch)
    return "".join(out)


def matlab_statements(text: str):
    """Split MATLAB source into statements: comments removed, `...` continuations joined,
    `a = 1; b = 2;` split on ';' outside brackets."""
    lines, buf = [], ""
    for raw in text.splitlines():
        code = _strip_comment(raw)
        if code.rstrip().endswith("..."):
            buf += code.rstrip()[:-3] + " "
            continue
        lines.append(buf + code)
        buf = ""
    if buf:
        lines.append(buf)
    for code in lines:
        depth, cur = 0, ""
        for ch in code:
            if ch in "[(":
                depth += 1
            elif ch in "])":
                depth -= 1
            if ch == ";" and depth == 0:
                if cur.strip():
                    yield cur.strip()
                cur = ""
            else:
                cur += ch
        if cur.strip():
            yield cur.strip()


# ---------------------------------------------------------------------------- #
# tiny safe evaluator for MATLAB-style numeric expressions
# ---------------------------------------------------------------------------- #
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_CONST = {"pi": math.pi, "nan": math.nan, "NaN": math.nan, "inf": math.inf, "Inf": math.inf,
          "true": 1.0, "false": 0.0}


def _eval(expr: str, env: dict[str, float]) -> float:
    e = expr.strip().rstrip(";")
    e = re.sub(r"\b[A-Za-z_]\w*\.([A-Za-z_]\w*)\b", r"\1", e)       # veh.lf -> lf (not 1.25)
    e = e.replace(".^", "**").replace("^", "**").replace(".*", "*").replace("./", "/")
    node = ast.parse(e, mode="eval").body

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name):
            if n.id in env:
                return env[n.id]
            if n.id in _CONST:
                return _CONST[n.id]
            raise NameError(n.id)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("sqrt", "deg2rad", "rad2deg"):
            a = ev(n.args[0])
            return {"sqrt": math.sqrt, "deg2rad": math.radians, "rad2deg": math.degrees}[n.func.id](a)
        raise ValueError("unsupported expression")

    return float(ev(node))
