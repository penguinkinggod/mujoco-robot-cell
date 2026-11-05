// glb2obj.mjs —— SimReady GLB -> OBJ（带物理数据导出 JSON）
//
// 为什么要走这一步
// ----------------
// SimReady 的 GLB 几何用了 EXT_meshopt_compression（meshopt 压缩），
// Python 的 trimesh 解不开（实测 IndexError）。
// glTF-Transform 是官方解码器，能正确解压，再导出 OBJ + 物理 JSON。
//
// 用法：
//   node glb2obj.mjs <input.glb> <outdir>
//
// 输出：
//   <outdir>/<name>.obj      三角网格
//   <outdir>/<name>.phys.json  质量/密度/碰撞凸包/关节

import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { MeshoptDecoder } from 'meshoptimizer';
import { writeFileSync, mkdirSync } from 'node:fs';
import { basename, join } from 'node:path';

// meshopt 必须先 await ready，否则解码时内部 wasm 还没初始化
await MeshoptDecoder.ready;

const [, , inPath, outDir] = process.argv;
if (!inPath || !outDir) {
  console.error('用法: node glb2obj.mjs <input.glb> <outdir>');
  process.exit(1);
}
mkdirSync(outDir, { recursive: true });

// 用 ALL_EXTENSIONS 把所有扩展都注册上，避免一个个补。
// 依赖键名必须是扩展内部写死的 'meshopt.decoder'（不是 'EXT_meshopt_compression'），
// 这个是从报错信息里读出来的，文档没写清楚。
const io = new NodeIO()
  .registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({
    'meshopt.decoder': MeshoptDecoder,
  });
const doc = await io.read(inPath);
const root = doc.getRoot();

const name = basename(inPath).replace(/\.glb$/i, '');

// ---------- 收集所有网格，转成 OBJ ----------
let objLines = [];
let vertBase = 1;          // OBJ 顶点索引从 1 开始
let totalVerts = 0;
let totalTris = 0;

function mat4ToRows(m) {
  // glTF 是列主序 16 个数，转成 4x4 行主序便于矩阵乘
  return [
    [m[0], m[4], m[8], m[12]],
    [m[1], m[5], m[9], m[13]],
    [m[2], m[6], m[10], m[14]],
    [m[3], m[7], m[11], m[15]],
  ];
}

// node 的变换：glTF-Transform 的 getMatrix() 已经等价于 TRS 展开后的矩阵。
// 实测（bench_vise）：getMatrix 和手算 TRS 得到完全相同的包围盒，
// 两者相乘则会重复缩放（0.246 -> 0.162）。所以只取 getMatrix()。
function nodeLocalMatrix(node) {
  return mat4ToRows(node.getMatrix());
}
function mul(a, b) {
  const r = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      for (let k = 0; k < 4; k++)
        r[i][j] += a[i][k] * b[k][j];
  return r;
}
function xform(m, v) {
  const x = m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2] + m[0][3];
  const y = m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2] + m[1][3];
  const z = m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2] + m[2][3];
  return [x, y, z];
}

const physNodes = [];

function walk(node, parentMatrix) {
  const local = nodeLocalMatrix(node);
  const world = mul(parentMatrix, local);

  const mesh = node.getMesh();
  if (mesh) {
    for (const prim of mesh.listPrimitives()) {
      const pos = prim.getAttribute('POSITION');
      if (!pos) continue;
      const idx = prim.getIndices();
      const nV = pos.getCount();
      const raw = pos.getArray();
      if (!raw) continue;

      // ★ 关键：KHR_mesh_quantization 把坐标编码成归一化整数，
      //   真实坐标 = raw * scale + offset（在 accessor 的 min/max 里定义）。
      //   直接读 raw 会得到 8800 这种数，看起来像毫米，其实是量化值。
      //   getElement() 会自动做反量化，所以用 API 逐点取，不手读 buffer。
      objLines.push(`o ${node.getName() || 'node'}_${totalVerts}`);
      const v = [0, 0, 0];
      for (let i = 0; i < nV; i++) {
        pos.getElement(i, v);
        const p = xform(world, [v[0], v[1], v[2]]);
        objLines.push(`v ${p[0].toFixed(6)} ${p[1].toFixed(6)} ${p[2].toFixed(6)}`);
      }
      if (idx) {
        const ia = idx.getArray();
        for (let i = 0; i < ia.length; i += 3) {
          objLines.push(`f ${ia[i]+vertBase} ${ia[i+1]+vertBase} ${ia[i+2]+vertBase}`);
        }
        totalTris += ia.length / 3;
      }
      totalVerts += nV;
      vertBase += nV;
    }
  }

  // 物理数据挂在 extras 上
  const ex = node.getExtras() || {};
  if (ex.mass !== undefined || ex.density !== undefined || ex.collisionHulls) {
    physNodes.push({
      name: node.getName(),
      mass: ex.mass ?? null,
      density: ex.density ?? null,
      joints: ex.joints ?? null,
      hulls: ex.collisionHulls ?? null,
      worldMatrix: world,
    });
  }

  for (const child of node.listChildren()) walk(child, world);
}

const identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
for (const scene of root.listScenes()) {
  for (const node of scene.listChildren()) walk(node, identity);
}

writeFileSync(join(outDir, `${name}.obj`), objLines.join('\n'), 'utf8');

writeFileSync(
  join(outDir, `${name}.phys.json`),
  JSON.stringify({ name, totalVerts, totalTris, nodes: physNodes }, null, 2),
  'utf8'
);

console.log(`OK ${name}: ${totalVerts} verts, ${totalTris} tris, ${physNodes.length} physics nodes`);
for (const p of physNodes) {
  const hullCount = p.hulls
    ? (Array.isArray(p.hulls) ? p.hulls.length : Object.keys(p.hulls).length)
    : 0;
  console.log(`   ${p.name}: mass=${p.mass} density=${p.density} hulls=${hullCount}`);
}
