const D = window.PYL209_DATA;
const N = D.n;
const N2 = N * N;
const SEG = D.segments;
const BLOCK = N / SEG;
const MODES = decodeFloat32(D.modes.data);
const TRANSVERSE_U = Float64Array.from(D.transverseU);
const form = document.getElementById("runForm");
const runButton = document.getElementById("runButton");
const presetButton = document.getElementById("presetButton");
const statusLine = document.getElementById("status");
const progressBar = document.getElementById("progressBar");
const stateBox = document.querySelector(".model-state");
const modelState = document.getElementById("modelState");
const downloadMask = document.getElementById("downloadMask");
const downloadResult = document.getElementById("downloadResult");
let W = null;
let lastResult = null;

document.getElementById("fibreLength").addEventListener("input", event => {
  document.getElementById("lengthValue").textContent = Number(event.target.value).toFixed(3) + " m";
});

document.getElementById("wavelength").addEventListener("input", event => {
  document.getElementById("wavelengthValue").textContent = Number(event.target.value).toFixed(0) + " nm";
});

form.addEventListener("submit", async event => {
  event.preventDefault();
  await runSimulation();
});

presetButton.addEventListener("click", async () => {
  document.getElementById("fibreLength").value = 1;
  document.getElementById("lengthValue").textContent = "1.000 m";
  document.getElementById("wavelength").value = 600;
  document.getElementById("wavelengthValue").textContent = "600 nm";
  document.getElementById("targetX").value = 0;
  document.getElementById("targetY").value = 0;
  document.getElementById("quality").value = "submitted";
  document.getElementById("seed").value = 209;
  await runSimulation();
});

function decodeFloat32(encoded) {
  const raw = atob(encoded);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function tensorFrom(name) {
  const item = D.weights[name];
  return tf.tensor(decodeFloat32(item.data), item.shape, "float32");
}

function transposed(name, order) {
  const source = tensorFrom(name);
  const value = source.transpose(order);
  source.dispose();
  return value;
}

async function initializeModel() {
  try {
    try { await tf.setBackend("webgl"); } catch (_) { await tf.setBackend("cpu"); }
    await tf.ready();
    W = {
      c0w: transposed("condition_encoder.0.weight", [2, 3, 1, 0]),
      c0b: tensorFrom("condition_encoder.0.bias"),
      c2w: transposed("condition_encoder.2.weight", [2, 3, 1, 0]),
      c2b: tensorFrom("condition_encoder.2.bias"),
      c5w: transposed("condition_encoder.5.weight", [1, 0]),
      c5b: tensorFrom("condition_encoder.5.bias"),
      diw: transposed("decoder_input.0.weight", [1, 0]),
      dib: tensorFrom("decoder_input.0.bias"),
      d0w: transposed("decoder.0.weight", [2, 3, 1, 0]),
      d0b: tensorFrom("decoder.0.bias"),
      d2w: transposed("decoder.2.weight", [2, 3, 1, 0]),
      d2b: tensorFrom("decoder.2.bias")
    };
    decodeBatch(new Float32Array(SEG * SEG * 5), new Float32Array(D.latentDim), 1);
    stateBox.classList.add("ready");
    modelState.textContent = `Trained decoder ready · ${tf.getBackend().toUpperCase()}`;
    statusLine.textContent = "Ready. Choose parameters or run the submitted case.";
    runButton.disabled = false;
    presetButton.disabled = false;
  } catch (error) {
    stateBox.classList.add("error");
    modelState.textContent = "Model could not be loaded";
    statusLine.textContent = error.message;
    statusLine.style.color = "#b3261e";
  }
}

function silu(value) {
  return value.mul(tf.sigmoid(value));
}

function decodeBatch(condition, latent, population) {
  return tf.tidy(() => {
    let value = tf.tensor4d(condition, [1, SEG, SEG, 5]).tile([population, 1, 1, 1]);
    value = tf.pad(value, [[0, 0], [1, 1], [1, 1], [0, 0]]);
    value = silu(tf.conv2d(value, W.c0w, 2, "valid").add(W.c0b));
    value = tf.pad(value, [[0, 0], [1, 1], [1, 1], [0, 0]]);
    value = silu(tf.conv2d(value, W.c2w, 2, "valid").add(W.c2b));
    value = value.transpose([0, 3, 1, 2]).reshape([population, 64 * 4 * 4]);
    value = silu(value.matMul(W.c5w).add(W.c5b));
    const latentTensor = tf.tensor2d(latent, [population, D.latentDim]);
    value = silu(tf.concat([value, latentTensor], 1).matMul(W.diw).add(W.dib));
    value = value.reshape([population, 64, 4, 4]).transpose([0, 2, 3, 1]);
    value = silu(tf.conv2dTranspose(value, W.d0w, [population, 8, 8, 32], 2, "same").add(W.d0b));
    value = tf.conv2dTranspose(value, W.d2w, [population, 16, 16, 2], 2, "same").add(W.d2b);
    value = value.div(value.square().sum(3, true).add(1e-8).sqrt());
    return Float32Array.from(value.dataSync());
  });
}

function phasesFromPrediction(prediction, population) {
  const result = new Float32Array(population * SEG * SEG);
  for (let i = 0; i < result.length; i++) result[i] = Math.atan2(prediction[2 * i + 1], prediction[2 * i]);
  return result;
}

function fft1(real, imag) {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let temp = real[i]; real[i] = real[j]; real[j] = temp;
      temp = imag[i]; imag[i] = imag[j]; imag[j] = temp;
    }
  }
  for (let length = 2; length <= n; length <<= 1) {
    const angle = -2 * Math.PI / length;
    const stepReal = Math.cos(angle);
    const stepImag = Math.sin(angle);
    for (let start = 0; start < n; start += length) {
      let wReal = 1;
      let wImag = 0;
      for (let offset = 0; offset < length / 2; offset++) {
        const left = start + offset;
        const right = left + length / 2;
        const vReal = real[right] * wReal - imag[right] * wImag;
        const vImag = real[right] * wImag + imag[right] * wReal;
        const uReal = real[left];
        const uImag = imag[left];
        real[left] = uReal + vReal; imag[left] = uImag + vImag;
        real[right] = uReal - vReal; imag[right] = uImag - vImag;
        const nextReal = wReal * stepReal - wImag * stepImag;
        wImag = wReal * stepImag + wImag * stepReal;
        wReal = nextReal;
      }
    }
  }
}

function fft2(real, imag) {
  const a = new Float64Array(N);
  const b = new Float64Array(N);
  for (let y = 0; y < N; y++) {
    const start = y * N;
    for (let x = 0; x < N; x++) { a[x] = real[start + x]; b[x] = imag[start + x]; }
    fft1(a, b);
    for (let x = 0; x < N; x++) { real[start + x] = a[x]; imag[start + x] = b[x]; }
  }
  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) { a[y] = real[y * N + x]; b[y] = imag[y * N + x]; }
    fft1(a, b);
    for (let y = 0; y < N; y++) { real[y * N + x] = a[y]; imag[y * N + x] = b[y]; }
  }
}

function centreShift(real, imag) {
  const shiftedReal = new Float64Array(N2);
  const shiftedImag = new Float64Array(N2);
  const half = N / 2;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const source = y * N + x;
    const target = ((y + half) % N) * N + ((x + half) % N);
    shiftedReal[target] = real[source];
    shiftedImag[target] = imag[source];
  }
  return {real: shiftedReal, imag: shiftedImag};
}

function launchCoefficients(wavelengthNm) {
  const real = new Float64Array(D.modeCount);
  const imag = new Float64Array(D.modeCount);
  const inputReal = new Float64Array(N2);
  const inputImag = new Float64Array(N2);
  const radius = D.coreRadiusM;
  const halfWidth = D.gridHalfWidthCoreRadii * radius;
  const wavelength = wavelengthNm * 1e-9;
  const waveNumber = 2 * Math.PI * D.nCore / wavelength;
  const angleX = 2.5 * Math.PI / 180;
  const angleY = -1.5 * Math.PI / 180;
  const offsetX = 9.0e-6;
  const offsetY = -4.5e-6;
  const waist = 2.2e-6;
  const vNumber = 2 * Math.PI * radius * D.numericalAperture / wavelength;
  let norm = 0;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const px = -halfWidth + 2 * halfWidth * x / (N - 1);
    const py = -halfWidth + 2 * halfWidth * y / (N - 1);
    const index = y * N + x;
    const amplitude = Math.exp(-((px - offsetX) ** 2 + (py - offsetY) ** 2) / (waist * waist));
    const phase = waveNumber * (Math.sin(angleX) * px + Math.sin(angleY) * py);
    inputReal[index] = amplitude * Math.cos(phase);
    inputImag[index] = amplitude * Math.sin(phase);
    norm += amplitude * amplitude;
  }
  norm = Math.sqrt(norm);
  for (let i = 0; i < N2; i++) { inputReal[i] /= norm; inputImag[i] /= norm; }
  let coefficientNorm = 0;
  for (let mode = 0; mode < D.modeCount; mode++) {
    if (TRANSVERSE_U[mode] >= vNumber) continue;
    const start = mode * N2;
    for (let i = 0; i < N2; i++) {
      real[mode] += MODES[start + i] * inputReal[i];
      imag[mode] += MODES[start + i] * inputImag[i];
    }
    coefficientNorm += real[mode] * real[mode] + imag[mode] * imag[mode];
  }
  coefficientNorm = Math.sqrt(coefficientNorm);
  for (let mode = 0; mode < D.modeCount; mode++) { real[mode] /= coefficientNorm; imag[mode] /= coefficientNorm; }
  return {real, imag};
}

function buildField(length, wavelengthNm) {
  const real = new Float64Array(N2);
  const imag = new Float64Array(N2);
  const coefficients = launchCoefficients(wavelengthNm);
  const wavelength = wavelengthNm * 1e-9;
  const k0 = 2 * Math.PI / wavelength;
  const radius = D.coreRadiusM;
  const beta0 = Math.sqrt((D.nCore * k0) ** 2 - (TRANSVERSE_U[0] / radius) ** 2);
  for (let mode = 0; mode < D.modeCount; mode++) {
    if (coefficients.real[mode] === 0 && coefficients.imag[mode] === 0) continue;
    const beta = Math.sqrt((D.nCore * k0) ** 2 - (TRANSVERSE_U[mode] / radius) ** 2);
    const phase = (beta - beta0) * length;
    const pReal = Math.cos(phase);
    const pImag = Math.sin(phase);
    const aReal = coefficients.real[mode] * pReal - coefficients.imag[mode] * pImag;
    const aImag = coefficients.real[mode] * pImag + coefficients.imag[mode] * pReal;
    const start = mode * N2;
    for (let i = 0; i < N2; i++) {
      const value = MODES[start + i];
      real[i] += aReal * value;
      imag[i] += aImag * value;
    }
  }
  let norm = 0;
  for (let i = 0; i < N2; i++) norm += real[i] * real[i] + imag[i] * imag[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < N2; i++) { real[i] /= norm; imag[i] /= norm; }
  const input = centreShift(real, imag);
  fft2(input.real, input.imag);
  const shifted = centreShift(input.real, input.imag);
  norm = 0;
  for (let i = 0; i < N2; i++) norm += shifted.real[i] * shifted.real[i] + shifted.imag[i] * shifted.imag[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < N2; i++) { shifted.real[i] /= norm; shifted.imag[i] /= norm; }
  return shifted;
}

function makeCondition(field, targetX, targetY) {
  const contributionReal = new Float64Array(SEG * SEG);
  const contributionImag = new Float64Array(SEG * SEG);
  for (let sy = 0; sy < SEG; sy++) for (let sx = 0; sx < SEG; sx++) {
    let sumReal = 0;
    let sumImag = 0;
    for (let by = 0; by < BLOCK; by++) for (let bx = 0; bx < BLOCK; bx++) {
      const y = sy * BLOCK + by;
      const x = sx * BLOCK + bx;
      const index = y * N + x;
      const axisX = x - N / 2;
      const axisY = y - N / 2;
      const angle = -2 * Math.PI * (targetX * axisX / N + targetY * axisY / N);
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      sumReal += field.real[index] * c - field.imag[index] * s;
      sumImag += field.real[index] * s + field.imag[index] * c;
    }
    const segment = sy * SEG + sx;
    contributionReal[segment] = sumReal;
    contributionImag[segment] = sumImag;
  }
  const amplitudes = Array.from(contributionReal, (value, index) => Math.hypot(value, contributionImag[index])).sort((a, b) => a - b);
  const scale = amplitudes[Math.floor(0.95 * (amplitudes.length - 1))] + 1e-8;
  const condition = new Float32Array(SEG * SEG * 5);
  for (let i = 0; i < SEG * SEG; i++) {
    const amplitude = Math.min(1, Math.hypot(contributionReal[i], contributionImag[i]) / scale);
    const phase = Math.atan2(contributionImag[i], contributionReal[i]);
    const start = 5 * i;
    condition[start] = amplitude;
    condition[start + 1] = Math.cos(phase);
    condition[start + 2] = Math.sin(phase);
    condition[start + 3] = targetX / 5;
    condition[start + 4] = targetY / 5;
  }
  return {condition, contributionReal, contributionImag};
}

function scorePhases(phases, population, optical) {
  const scores = new Float64Array(population);
  for (let candidate = 0; candidate < population; candidate++) {
    let real = 0;
    let imag = 0;
    const start = candidate * SEG * SEG;
    for (let i = 0; i < SEG * SEG; i++) {
      const angle = phases[start + i];
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      real += optical.contributionReal[i] * c - optical.contributionImag[i] * s;
      imag += optical.contributionReal[i] * s + optical.contributionImag[i] * c;
    }
    const target = Math.min(0.999999999, (real * real + imag * imag) / N2);
    scores[candidate] = target / ((1 - target) / (N2 - 1) + 1e-12);
  }
  return scores;
}

function phaseAt(phases, index) {
  return phases.slice(index * SEG * SEG, (index + 1) * SEG * SEG);
}

function indexOfMax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function normal(rng) {
  const u = Math.max(1e-12, rng());
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function latentSamples(rng, population, mean, std) {
  const values = new Float32Array(population * D.latentDim);
  for (let p = 0; p < population; p++) for (let j = 0; j < D.latentDim; j++) values[p * D.latentDim + j] = mean[j] + std[j] * normal(rng);
  return values;
}

function randomPhases(rng, population) {
  const phases = new Float32Array(population * SEG * SEG);
  for (let i = 0; i < phases.length; i++) phases[i] = -Math.PI + 2 * Math.PI * rng();
  return phases;
}

function focusImage(field, phase) {
  const real = new Float64Array(N2);
  const imag = new Float64Array(N2);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const index = y * N + x;
    const angle = phase[Math.floor(y / BLOCK) * SEG + Math.floor(x / BLOCK)];
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    real[index] = field.real[index] * c - field.imag[index] * s;
    imag[index] = field.real[index] * s + field.imag[index] * c;
  }
  const input = centreShift(real, imag);
  fft2(input.real, input.imag);
  const shifted = centreShift(input.real, input.imag);
  const intensity = new Float64Array(N2);
  let maximum = 0;
  for (let i = 0; i < N2; i++) {
    const value = shifted.real[i] * shifted.real[i] + shifted.imag[i] * shifted.imag[i];
    intensity[i] = value;
    if (value > maximum) maximum = value;
  }
  const image = Array.from({length: N}, () => Array(N));
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const value = intensity[y * N + x] / (maximum + 1e-20);
    image[y][x] = Math.max(0, Math.min(1, (Math.log10(value + 1e-6) + 6) / 6));
  }
  return image;
}

async function runSimulation() {
  if (!W) return;
  setBusy(true);
  clearResults();
  try {
    const length = Number(document.getElementById("fibreLength").value);
    const wavelength = Number(document.getElementById("wavelength").value);
    const targetX = Number(document.getElementById("targetX").value);
    const targetY = Number(document.getElementById("targetY").value);
    const quality = document.getElementById("quality").value;
    const seed = Number(document.getElementById("seed").value);
    const settings = quality === "quick" ? {population: 32, rounds: 0} : quality === "balanced" ? {population: 64, rounds: 8} : {population: 96, rounds: 20};
    const rng = makeRng(seed);

    statusLine.textContent = "Generating the multimode field…";
    progressBar.style.width = "8%";
    await tf.nextFrame();
    const field = buildField(length, wavelength);
    const optical = makeCondition(field, targetX, targetY);

    const zeroPhase = new Float32Array(SEG * SEG);
    const idealPhase = new Float32Array(SEG * SEG);
    for (let i = 0; i < idealPhase.length; i++) idealPhase[i] = -Math.atan2(optical.contributionImag[i], optical.contributionReal[i]);
    const randomSet = randomPhases(rng, settings.population);
    const randomScores = scorePhases(randomSet, settings.population, optical);
    const bestRandom = phaseAt(randomSet, indexOfMax(randomScores));
    const zeroScore = scorePhases(zeroPhase, 1, optical)[0];
    const boundScore = scorePhases(idealPhase, 1, optical)[0];

    statusLine.textContent = "Generating cVAE candidates…";
    progressBar.style.width = "22%";
    await tf.nextFrame();
    const mean = new Float32Array(D.latentDim);
    const std = new Float32Array(D.latentDim).fill(1);
    let latent = latentSamples(rng, settings.population, mean, std);
    let prediction = decodeBatch(optical.condition, latent, settings.population);
    let phases = phasesFromPrediction(prediction, settings.population);
    let scores = scorePhases(phases, settings.population, optical);
    let directScore = scores[indexOfMax(scores)];
    const directPhase = phaseAt(phases, indexOfMax(scores));
    let refinedPhase = directPhase.slice();
    let refinedScore = directScore;
    const trace = [];

    for (let round = 0; round < settings.rounds; round++) {
      latent = latentSamples(rng, settings.population, mean, std);
      prediction = decodeBatch(optical.condition, latent, settings.population);
      phases = phasesFromPrediction(prediction, settings.population);
      scores = scorePhases(phases, settings.population, optical);
      const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);
      if (scores[order[0]] > refinedScore) {
        refinedScore = scores[order[0]];
        refinedPhase = phaseAt(phases, order[0]);
      }
      const eliteCount = Math.max(4, Math.min(16, Math.floor(settings.population / 8)));
      for (let j = 0; j < D.latentDim; j++) {
        let eliteMean = 0;
        for (let e = 0; e < eliteCount; e++) eliteMean += latent[order[e] * D.latentDim + j];
        eliteMean /= eliteCount;
        let variance = 0;
        for (let e = 0; e < eliteCount; e++) variance += (latent[order[e] * D.latentDim + j] - eliteMean) ** 2;
        const eliteStd = Math.sqrt(variance / eliteCount);
        mean[j] = 0.35 * mean[j] + 0.65 * eliteMean;
        std[j] = Math.max(0.06, 0.35 * std[j] + 0.65 * eliteStd);
      }
      trace.push(refinedScore);
      progressBar.style.width = `${30 + 55 * (round + 1) / settings.rounds}%`;
      statusLine.textContent = `Refinement round ${round + 1}/${settings.rounds}; best PBR ${refinedScore.toFixed(2)}.`;
      drawTrace(trace);
      await tf.nextFrame();
    }

    if (!trace.length) trace.push(directScore);
    statusLine.textContent = "Rendering comparison images…";
    progressBar.style.width = "90%";
    await tf.nextFrame();
    const images = {
      uncorrected: focusImage(field, zeroPhase),
      random: focusImage(field, bestRandom),
      direct: focusImage(field, directPhase),
      refined: focusImage(field, refinedPhase),
      bound: focusImage(field, idealPhase)
    };
    const result = {
      parameters: {fibre_length: length, wavelength_nm: wavelength, target_x: targetX, target_y: targetY, quality, seed, population: settings.population, rounds: settings.rounds},
      backend: tf.getBackend(),
      scores: {
        "No correction": zeroScore,
        [`Best of ${settings.population} random`]: randomScores[indexOfMax(randomScores)],
        [`Best of ${settings.population} cVAE`]: directScore,
        "cVAE with refinement": refinedScore,
        "Phase-conjugate bound": boundScore
      },
      trace,
      phase_mask: rowsFromPhase(refinedPhase),
      images
    };
    showResult(result);
  } catch (error) {
    statusLine.textContent = error.message;
    statusLine.style.color = "#b3261e";
  } finally {
    setBusy(false);
  }
}

function rowsFromPhase(phase) {
  return Array.from({length: SEG}, (_, y) => Array.from(phase.slice(y * SEG, (y + 1) * SEG), value => Number(value.toFixed(6))));
}

function showResult(result) {
  lastResult = result;
  progressBar.style.width = "100%";
  statusLine.style.color = "";
  statusLine.textContent = `Completed locally at ${result.parameters.wavelength_nm} nm using ${result.backend.toUpperCase()}; seed ${result.parameters.seed}.`;
  const entries = Object.entries(result.scores);
  const direct = entries.find(([name]) => name.includes("cVAE") && !name.includes("refinement"));
  const refined = entries.find(([name]) => name.includes("refinement"));
  const bound = entries.find(([name]) => name.includes("bound"));
  const cards = document.querySelectorAll("#scoreCards strong");
  cards[0].textContent = direct[1].toFixed(2);
  cards[1].textContent = refined[1].toFixed(2);
  cards[2].textContent = bound[1].toFixed(2);
  document.getElementById("scoreCards").classList.remove("empty");
  drawPhase(document.getElementById("maskCanvas"), result.phase_mask);
  drawHeat(document.getElementById("focusCanvas"), result.images.refined);
  drawTrace(result.trace);
  document.querySelectorAll("[data-image]").forEach(canvas => drawHeat(canvas, result.images[canvas.dataset.image]));
  document.getElementById("scoreTable").innerHTML = entries.map(([name, value]) => `<tr><td>${escapeHtml(name)}</td><td>${value.toFixed(3)}</td></tr>`).join("");
  downloadMask.disabled = false;
  downloadResult.disabled = false;
}

function drawPhase(canvas, values) {
  const n = values.length;
  const small = document.createElement("canvas"); small.width = n; small.height = n;
  const context = small.getContext("2d"); const image = context.createImageData(n, n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const hue = (values[y][x] + Math.PI) / (2 * Math.PI);
    const [r, g, b] = hslToRgb(hue, 0.78, 0.52); const i = 4 * (y * n + x);
    image.data.set([r, g, b, 255], i);
  }
  context.putImageData(image, 0, 0); const target = canvas.getContext("2d"); target.imageSmoothingEnabled = false; target.drawImage(small, 0, 0, canvas.width, canvas.height);
}

function drawHeat(canvas, values) {
  const n = values.length;
  const small = document.createElement("canvas"); small.width = n; small.height = n;
  const context = small.getContext("2d"); const image = context.createImageData(n, n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const [r, g, b] = inferno(values[y][x]); const i = 4 * (y * n + x); image.data.set([r, g, b, 255], i);
  }
  context.putImageData(image, 0, 0); const target = canvas.getContext("2d"); target.imageSmoothingEnabled = false; target.drawImage(small, 0, 0, canvas.width, canvas.height);
}

function drawTrace(trace) {
  const canvas = document.getElementById("traceCanvas"); const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height); context.fillStyle = "#f8fafc"; context.fillRect(0, 0, canvas.width, canvas.height);
  const left = 46, top = 18, width = canvas.width - 62, height = canvas.height - 54;
  context.strokeStyle = "#c8d0dc"; context.strokeRect(left, top, width, height);
  if (!trace.length) return;
  const maximum = Math.max(...trace) * 1.08;
  context.beginPath(); context.strokeStyle = "#174ea6"; context.lineWidth = 3;
  trace.forEach((value, index) => { const x = left + (trace.length === 1 ? 0.5 : index / (trace.length - 1)) * width; const y = top + (1 - value / maximum) * height; index ? context.lineTo(x, y) : context.moveTo(x, y); }); context.stroke();
  context.fillStyle = "#4f5b6d"; context.font = "13px system-ui"; context.fillText("PBR", 8, 26); context.fillText("refinement round", canvas.width / 2 - 50, canvas.height - 8); context.fillText(maximum.toFixed(1), 7, 22); context.fillText("0", 25, top + height + 4);
}

function inferno(t) { t = Math.max(0, Math.min(1, t)); return [Math.round(255 * Math.min(1, 1.8 * t)), Math.round(255 * Math.max(0, Math.min(1, 1.9 * t - 0.55))), Math.round(255 * Math.max(0, Math.min(1, 2.6 * t - 1.55)))]; }
function hslToRgb(h, s, l) { let r, g, b; if (!s) r = g = b = l; else { const f = (p, q, t) => { if (t < 0) t += 1; if (t > 1) t -= 1; if (t < 1 / 6) return p + (q - p) * 6 * t; if (t < 1 / 2) return q; if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6; return p; }; const q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q; r = f(p, q, h + 1 / 3); g = f(p, q, h); b = f(p, q, h - 1 / 3); } return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]; }
function setBusy(value) { runButton.disabled = value; presetButton.disabled = value; runButton.textContent = value ? "Calculating…" : "Generate focus"; }
function clearResults() { lastResult = null; progressBar.style.width = "0"; downloadMask.disabled = true; downloadResult.disabled = true; statusLine.style.color = ""; document.getElementById("scoreCards").classList.add("empty"); }
function escapeHtml(value) { return value.replace(/[&<>'"]/g, character => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"}[character])); }
function saveFile(name, text, type) { const blob = new Blob([text], {type}); const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 0); }
downloadMask.addEventListener("click", () => { if (lastResult) saveFile("refined_phase_mask.csv", lastResult.phase_mask.map(row => row.join(",")).join("\n"), "text/csv"); });
downloadResult.addEventListener("click", () => { if (lastResult) saveFile("part7b_browser_result.json", JSON.stringify(lastResult, null, 2), "application/json"); });

initializeModel();
