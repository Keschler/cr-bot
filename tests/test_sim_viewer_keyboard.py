"""Playback must keep stepping after controls receive keyboard focus."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_playback_shortcuts_respect_focus_and_native_form_controls():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the native JavaScript shortcut regression")
    module = (Path(__file__).resolve().parents[1] / "src/sim_viewer/static/keyboard.js").as_uri()
    script = r"""
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source = readFileSync(new URL(process.argv[1]), 'utf8');
const {playbackCommand} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
function target(tag, props = {}) {
  return {...props, closest(selector) {
    return selector.split(',').map(s => s.trim()).includes(tag) ? this : null;
  }};
}
const command = (key, element, extra = {}, context) => playbackCommand({key, target: element, ...extra}, context);
// Clicking a button or its icon leaves it focused; arrows must still step.
for (const button of [target('button'), {closest: s => s === 'button' ? {} : null}]) {
  assert.deepEqual(command('ArrowRight', button), {action: 'step', delta: 1});
  assert.deepEqual(command('ArrowLeft', button), {action: 'step', delta: -1});
  assert.deepEqual(command('ArrowRight', button, {shiftKey: true}), {action: 'step', delta: 10});
  assert.equal(command(' ', button, {code: 'Space'}), null);
}
const scrubber = target('input', {id: 'scrubber', type: 'range'});
assert.deepEqual(command('ArrowLeft', scrubber), {action: 'step', delta: -1});
assert.deepEqual(command('Home', scrubber), {action: 'first'});
assert.deepEqual(command('End', scrubber), {action: 'last'});
const overlay = target('input', {type: 'checkbox'});
assert.deepEqual(command('ArrowRight', overlay), {action: 'step', delta: 1});
assert.equal(command(' ', overlay, {code: 'Space'}), null);
for (const element of [target('input', {type: 'search'}), target('input', {type: 'number'}), target('textarea'), target('select'), target('span', {isContentEditable: true})]) {
  for (const key of ['ArrowLeft', 'ArrowRight', 'Home', 'End', 'l', ' ']) assert.equal(command(key, element), null);
}
for (const props of [{ctrlKey: true}, {metaKey: true}, {altKey: true}, {isComposing: true}, {defaultPrevented: true}]) {
  assert.equal(command('ArrowRight', target('canvas'), props), null);
}
assert.equal(command('ArrowRight', target('button'), {}, {page: 'training'}), null);
assert.equal(command('ArrowRight', target('button'), {}, {dialogOpen: true}), null);
assert.deepEqual(command(' ', target('canvas'), {code: 'Space'}), {action: 'play'});
"""
    result = subprocess.run([node, "--input-type=module", "-e", script, module], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
