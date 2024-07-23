export default function keyMapper(key: string, base: number) {
    const keyMap: { [key: string]: number } = {
      KeyA: 0,
      KeyW: 1,
      KeyS: 2,
      KeyE: 3,
      KeyD: 4,
      KeyF: 5,
      KeyT: 6,
      KeyG: 7,
      KeyY: 8,
      KeyH: 9,
      KeyU: 10,
      KeyJ: 11,
      KeyK: 12,
      KeyO: 13,
      KeyL: 14,
      KeyP: 15,
    };
  
    if (key in keyMap) {
      return base + keyMap[key];
    }
  }
  