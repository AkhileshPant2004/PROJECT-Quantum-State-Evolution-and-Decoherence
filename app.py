from flask import Flask, render_template, request, jsonify, send_from_directory, abort
import os
import time
import numpy as np
import cloudpickle as pickle
from qutip import (basis, tensor, ket2dm, fidelity, concurrence,
                   destroy, sigmax, sigmaz, qeye, mesolve)

# -------------------------------
# App setup
# -------------------------------
# Flask looks for templates in a folder literally named "templates" next to
# this file by default. That folder (and index.html inside it) was missing,
# which is exactly what caused the TemplateNotFound error. template_folder
# is set explicitly here too, just to make the expected layout obvious:
#
#   QuantumState/
#     app.py
#     templates/
#       index.html      <-- the dashboard from our chat
#     simulations/       <-- created automatically, stores .pkl results
#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
SIM_DIR = os.path.join(BASE_DIR, 'simulations')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
os.makedirs(SIM_DIR, exist_ok=True)


# -------------------------------
# Simulation helper functions
# -------------------------------
def create_single_qubit(state="plus"):
    if state == "zero":
        return basis(2, 0)
    elif state == "one":
        return basis(2, 1)
    elif state == "plus":
        return (basis(2, 0) + basis(2, 1)).unit()
    elif state == "minus":
        return (basis(2, 0) - basis(2, 1)).unit()
    return basis(2, 0)


def create_bell_pair():
    q0 = basis(2, 0)
    q1 = basis(2, 1)
    return (tensor(q0, q0) + tensor(q1, q1)).unit()


# Noise models
def apply_phase_noise(rho, gamma=0.1, tlist=[0, 1]):
    sz1 = tensor(sigmaz(), qeye(2))
    sz2 = tensor(qeye(2), sigmaz())
    L = [gamma * sz1, gamma * sz2]
    result = mesolve(H=0 * rho, rho0=rho, tlist=tlist, c_ops=L)
    return result.states[-1]


def apply_amplitude_damping(rho, gamma=0.1, tlist=[0, 1]):
    c_ops = [gamma * tensor(destroy(2), qeye(2)), gamma * tensor(qeye(2), destroy(2))]
    result = mesolve(H=0 * rho, rho0=rho, tlist=tlist, c_ops=c_ops)
    return result.states[-1]


def apply_depolarizing_noise(rho, p=0.1):
    I = qeye(2)
    return (1 - p) * rho + (p / 4) * tensor(I, I)


def apply_bit_flip(rho, p=0.1):
    X = sigmax()
    return (1 - p) * rho + p * tensor(X, X) * rho * tensor(X, X)


def apply_phase_flip(rho, p=0.1):
    Z = sigmaz()
    return (1 - p) * rho + p * tensor(Z, Z) * rho * tensor(Z, Z)


# Measurements
def compute_probabilities(rho):
    rho_q0 = rho.ptrace(0)
    rho_q1 = rho.ptrace(1)
    return {
        "q0_0": float(rho_q0.diag()[0].real),
        "q0_1": float(rho_q0.diag()[1].real),
        "q1_0": float(rho_q1.diag()[0].real),
        "q1_1": float(rho_q1.diag()[1].real)
    }


def compute_fidelity(rho, rho_target):
    return float(fidelity(rho, rho_target))


def compute_entanglement(rho):
    try:
        return float(concurrence(rho))
    except Exception:
        return 0.0


def compute_purity(rho):
    # Tr(rho^2) -- this is shown on the dashboard as "Purity" but the
    # original backend never computed it, so the frontend's metric card
    # would otherwise have nothing real to display from a live run.
    return float((rho * rho).tr().real)


# Simulation runner
def run_simulation(noise_type, gamma, p, t_steps):
    bell_state = create_bell_pair()
    bell_density = ket2dm(bell_state)
    # ensure at least two points in tlist
    length = max(int(t_steps), 2)
    tlist = np.linspace(0, max(1, t_steps), length)

    prob_over_time = []
    fidelity_over_time = []
    entanglement_over_time = []
    purity_over_time = []

    for t in tlist:
        if noise_type == "Phase Noise":
            rho = apply_phase_noise(bell_density, gamma=gamma, tlist=[0, t])
        elif noise_type == "Amplitude Damping":
            rho = apply_amplitude_damping(bell_density, gamma=gamma, tlist=[0, t])
        elif noise_type == "Depolarizing Noise":
            rho = apply_depolarizing_noise(bell_density, p=p)
        elif noise_type == "Bit Flip":
            rho = apply_bit_flip(bell_density, p=p)
        elif noise_type == "Phase Flip":
            rho = apply_phase_flip(bell_density, p=p)
        else:
            rho = bell_density

        prob_over_time.append(compute_probabilities(rho))
        fidelity_over_time.append(compute_fidelity(rho, bell_density))
        entanglement_over_time.append(compute_entanglement(rho))
        purity_over_time.append(compute_purity(rho))

    return {
        'tlist': tlist.tolist(),
        'prob_time': prob_over_time,
        'fidelity_time': fidelity_over_time,
        'ent_time': entanglement_over_time,
        'purity_time': purity_over_time
    }


# -------------------------------
# Flask routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run', methods=['POST'])
def run():
    payload = request.get_json(force=True)
    noise_type = payload.get('noise_type', 'Phase Noise')
    gamma = float(payload.get('gamma', 0.05))
    p = float(payload.get('p', 0.05))
    t_steps = int(payload.get('t_steps', 50))

    t0 = time.perf_counter()
    sim = run_simulation(noise_type, gamma, p, t_steps)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Save simulation to file
    safe_name = noise_type.replace(' ', '_')
    filename = f"quantum_simulation_{safe_name}_g{gamma}_p{p}_t{t_steps}.pkl"
    filepath = os.path.join(SIM_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump({
            'params': {'noise_type': noise_type, 'gamma': gamma, 'p': p, 't_steps': t_steps},
            'result': sim
        }, f)

    # Field names below (params/result/tlist/prob_time/fidelity_time/
    # ent_time/purity_time) match what index.html's downloadJson /
    # downloadCsv handlers and the metric cards expect, so the same
    # template can be pointed at this real QuTiP-backed endpoint instead
    # of its built-in client-side math if you wire up a fetch() call to
    # /run from the "Run Simulation" button.
    response = {
        'filename': filename,
        'params': {'noise_type': noise_type, 'gamma': gamma, 'p': p, 't_steps': t_steps},
        'summary': {
            'final_fidelity': sim['fidelity_time'][-1],
            'final_entanglement': sim['ent_time'][-1],
            'final_purity': sim['purity_time'][-1],
            'elapsed_ms': elapsed_ms
        },
        'result': sim
    }
    return jsonify(response)


@app.route('/download/<path:filename>')
def download_file(filename):
    safe_path = os.path.join(SIM_DIR, filename)
    if not os.path.exists(safe_path):
        abort(404)
    return send_from_directory(SIM_DIR, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)