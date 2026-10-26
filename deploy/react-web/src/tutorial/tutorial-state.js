import lessons from "../lib/tutorial-lessons.js";
import { getFrameCount } from "./canvas-engine.js";

function defaults(lesson) {
  return Object.fromEntries(lesson.controls.map(control => [control.id, control.value]));
}

export function createTutorialState(id = "", stored = "[]") {
  const selected = Math.max(0, lessons.findIndex(lesson => lesson.id === id));
  let saved = [];
  try { const parsed = JSON.parse(stored); if (Array.isArray(parsed)) saved = parsed; } catch { saved = []; }
  const visited = [...new Set(saved.filter(key => lessons.some(lesson => lesson.id === key)).concat(lessons[selected].id))];
  return { selected, parameters: defaults(lessons[selected]), frame: 0, fraction: 0, playing: false, speed: 1050, answer: null, visited, selectionVersion: 0 };
}

export function tutorialReducer(state, action) {
  const lesson = lessons[state.selected];
  const frames = getFrameCount(lesson, state.parameters);
  switch (action.type) {
    case "choose": {
      const selected = Math.max(0, Math.min(lessons.length - 1, action.index));
      return { ...state, selected, parameters: defaults(lessons[selected]), frame: 0, fraction: 0, playing: false, answer: null, visited: [...new Set(state.visited.concat(lessons[selected].id))], selectionVersion: state.selectionVersion + 1 };
    }
    case "control":
      return { ...state, parameters: { ...state.parameters, [action.id]: action.value }, frame: 0, fraction: 0, playing: false };
    case "play":
      return state.playing ? { ...state, playing: false } : { ...state, playing: true, frame: state.frame >= frames - 1 ? 0 : state.frame, fraction: 0 };
    case "pause":
      return { ...state, playing: false };
    case "step":
      return { ...state, playing: false, fraction: 0, frame: Math.min(frames - 1, state.frame + 1) };
    case "restart":
      return { ...state, playing: false, frame: 0, fraction: 0 };
    case "advance":
      return state.frame >= frames - 1 ? { ...state, playing: false, fraction: 0 } : { ...state, frame: state.frame + 1, fraction: 0 };
    case "fraction":
      return { ...state, fraction: action.value };
    case "speed":
      return { ...state, speed: Number(action.value) };
    case "answer":
      return { ...state, answer: action.index };
    default:
      return state;
  }
}
