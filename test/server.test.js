import test from 'node:test';
import assert from 'node:assert/strict';
import { extractScores, validateUpload } from '../server.js';

const pngBuffer = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  'base64'
);

test('extractScores normalizes valid model labels and ignores malformed items', () => {
  const result = extractScores([
    { label: 'AI-generated', score: 0.8 },
    { label: 'real', score: 0.2 },
    { label: null, score: 'invalid' },
  ]);

  assert.equal(result.fakeScore, 0.8);
  assert.equal(result.realScore, 0.2);
  assert.equal(result.raw.length, 2);
});

test('validateUpload accepts a real PNG signature', async () => {
  const error = await validateUpload(
    { buffer: pngBuffer },
    new Set(['image/png'])
  );

  assert.equal(error, null);
});

test('validateUpload rejects bytes with an unsupported signature', async () => {
  const error = await validateUpload(
    { buffer: Buffer.from('not an image') },
    new Set(['image/png'])
  );

  assert.equal(error, 'Unsupported or invalid file type.');
});
