export function depthSortedOrder(xyz, matrixElements) {
  const count = ArrayBuffer.isView(xyz) ? xyz.length / 3 : xyz.length;
  const depths = new Float32Array(count);
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i += 1) {
    const x = ArrayBuffer.isView(xyz) ? xyz[i * 3] : xyz[i][0];
    const y = ArrayBuffer.isView(xyz) ? xyz[i * 3 + 1] : xyz[i][1];
    const z = ArrayBuffer.isView(xyz) ? xyz[i * 3 + 2] : xyz[i][2];
    depths[i] = matrixElements[2] * x + matrixElements[6] * y + matrixElements[10] * z + matrixElements[14];
    order[i] = i;
  }
  order.sort((a, b) => depths[a] - depths[b]);
  return order;
}

export function reorderByOrder(values, order, components = 1) {
  if (ArrayBuffer.isView(values)) {
    const sorted = new values.constructor(order.length * components);
    for (let i = 0; i < order.length; i += 1) {
      const source = order[i] * components;
      const target = i * components;
      for (let component = 0; component < components; component += 1) {
        sorted[target + component] = values[source + component];
      }
    }
    return sorted;
  }
  const sorted = new Array(order.length);
  for (let i = 0; i < order.length; i += 1) sorted[i] = values[order[i]];
  return sorted;
}
