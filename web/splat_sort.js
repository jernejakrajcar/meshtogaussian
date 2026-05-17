export function depthSortedOrder(xyz, matrixElements) {
  const depths = new Float32Array(xyz.length);
  const order = new Uint32Array(xyz.length);
  for (let i = 0; i < xyz.length; i += 1) {
    const point = xyz[i];
    depths[i] = matrixElements[2] * point[0] + matrixElements[6] * point[1] + matrixElements[10] * point[2] + matrixElements[14];
    order[i] = i;
  }
  order.sort((a, b) => depths[a] - depths[b]);
  return order;
}

export function reorderByOrder(values, order) {
  const sorted = new Array(order.length);
  for (let i = 0; i < order.length; i += 1) sorted[i] = values[order[i]];
  return sorted;
}
