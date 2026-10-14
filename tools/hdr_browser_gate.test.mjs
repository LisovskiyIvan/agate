import test from 'node:test';
import assert from 'node:assert/strict';
import { compareCoarse, isInformationalDiagnostic } from './hdr_browser_gate.mjs';

const image = (rgba, width = 1) => ({ width, height: rgba.length / (width * 4), px: Buffer.from(rgba) });

test('disabled-path parity compares every RGBA channel', () => {
  const original = image([24, 48, 96, 255]);
  const copy = image([24, 48, 96, 255]);
  const same = compareCoarse(original, copy);
  assert.equal(same.changedPixels, 0);
  assert.equal(same.rmseRgb, 0);
  const alphaOnly = compareCoarse(original, image([24, 48, 96, 254]));
  assert.equal(alphaOnly.meanAbs, 0); // Luma equality is not pixel parity.
  assert.equal(alphaOnly.changedPixels, 1);
  assert.equal(alphaOnly.maxChannelAbs, 1 / 255);
});

test('RGB RMSE and changed pixels are normalized per image', () => {
  const a = image([0, 0, 0, 255, 0, 0, 0, 255], 2);
  const b = image([255, 0, 0, 255, 0, 0, 0, 255], 2);
  const result = compareCoarse(a, b);
  assert.equal(result.changedPixels, 1);
  assert.equal(result.maxChannelAbs, 1);
  assert.ok(Math.abs(result.rmseRgb - Math.sqrt(1 / 6)) < 1e-12);
  assert.ok(Math.abs(result.meanAbs - 0.2126 / 2) < 1e-12);
});

test('different canvas sizes cannot pass image comparison', () => {
  assert.deepEqual(compareCoarse(image([0, 0, 0, 255]), image([0, 0, 0, 255, 0, 0, 0, 255])), { sameSize: false });
});

test('finite HDR progress on stderr is not a rendering failure', () => {
  assert.ok(isInformationalDiagnostic('hdr-showcase: render target 1280x720 scale=1.00 swapchain 1280x720'));
  assert.ok(isInformationalDiagnostic('phase 0: scale=1.00 no samples (occl 0) target=1280x720 window=1280x720'));
  assert.ok(isInformationalDiagnostic('q0-interior: frame30 draws=10 post_draws=4', true));
  assert.equal(isInformationalDiagnostic('q0-interior: frame30 draws=10 post_draws=4'), false);
});

test('known progress prefixes never hide hard failures', () => {
  assert.equal(isInformationalDiagnostic('hdr-showcase: render target 1280x720 scale=1.00 swapchain 1280x720 GPUValidationError'), false);
  assert.equal(isInformationalDiagnostic('q0-interior: panic in renderer', true), false);
  assert.equal(isInformationalDiagnostic('hdr-showcase: FAIL check'), false);
  assert.equal(isInformationalDiagnostic('unknown console.error'), false);
});
