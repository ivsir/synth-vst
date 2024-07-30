import PianoKeyboard from "./panels/pianoKeyboard";
import ChordProgressions from "./panels/chords";

export default function index() {
  return /*html*/ `
  <div class="synthContainer">
      ${PianoKeyboard()}
      ${ChordProgressions()}
  </div>
  `;
}
