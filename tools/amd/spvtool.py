"""Small SPIR-V reader for the AMD pipeline investigation (no SPIRV-Tools needed).

  python tools/amd/spvtool.py dis  <file.spv> [-o out.txt]   disassembly (ids as %N, OpName strings shown)
  python tools/amd/spvtool.py stats <file.spv> [...]          per-module and per-function statistics
  python tools/amd/spvtool.py funcs <file.spv>                functions with their size, blocks, calls and callers

It reads modules dumped by the runtime (OOT_PIPE_DUMP, OOT_DUMP_CS_WORDS, OOT_DUMP_MODULE_WORDS); those hold the
game's shaders and stay local.
The opcode table covers the core instructions the translator emits; anything else prints as Op<number>.
"""
import struct
import sys
from collections import Counter, defaultdict

OPS = {
    0: "Nop", 1: "Undef", 3: "Source", 4: "SourceExtension", 5: "Name", 6: "MemberName", 7: "String", 8: "Line",
    10: "Extension", 11: "ExtInstImport", 12: "ExtInst", 14: "MemoryModel", 15: "EntryPoint", 16: "ExecutionMode",
    17: "Capability", 19: "TypeVoid", 20: "TypeBool", 21: "TypeInt", 22: "TypeFloat", 23: "TypeVector",
    24: "TypeMatrix", 25: "TypeImage", 26: "TypeSampler", 27: "TypeSampledImage", 28: "TypeArray",
    29: "TypeRuntimeArray", 30: "TypeStruct", 32: "TypePointer", 33: "TypeFunction", 39: "TypeForwardPointer",
    41: "ConstantTrue", 42: "ConstantFalse", 43: "Constant", 44: "ConstantComposite", 46: "ConstantNull",
    48: "SpecConstantTrue", 49: "SpecConstantFalse", 50: "SpecConstant", 51: "SpecConstantComposite",
    52: "SpecConstantOp", 54: "Function", 55: "FunctionParameter", 56: "FunctionEnd", 57: "FunctionCall",
    59: "Variable", 60: "ImageTexelPointer", 61: "Load", 62: "Store", 63: "CopyMemory", 65: "AccessChain",
    66: "InBoundsAccessChain", 67: "PtrAccessChain", 68: "ArrayLength", 71: "Decorate", 72: "MemberDecorate",
    77: "VectorExtractDynamic", 78: "VectorInsertDynamic", 79: "VectorShuffle", 80: "CompositeConstruct",
    81: "CompositeExtract", 82: "CompositeInsert", 83: "CopyObject", 84: "Transpose", 86: "SampledImage",
    87: "ImageSampleImplicitLod", 88: "ImageSampleExplicitLod", 89: "ImageSampleDrefImplicitLod",
    90: "ImageSampleDrefExplicitLod", 91: "ImageSampleProjImplicitLod", 92: "ImageSampleProjExplicitLod",
    93: "ImageSampleProjDrefImplicitLod", 94: "ImageSampleProjDrefExplicitLod", 95: "ImageFetch", 96: "ImageGather",
    97: "ImageDrefGather", 98: "ImageRead", 99: "ImageWrite", 100: "Image", 103: "ImageQuerySizeLod",
    104: "ImageQuerySize", 105: "ImageQueryLod", 106: "ImageQueryLevels", 107: "ImageQuerySamples",
    109: "ConvertFToU", 110: "ConvertFToS", 111: "ConvertSToF", 112: "ConvertUToF", 113: "UConvert", 114: "SConvert",
    115: "FConvert", 116: "QuantizeToF16", 117: "ConvertPtrToU", 120: "ConvertUToPtr", 124: "Bitcast", 126: "SNegate",
    127: "FNegate", 128: "IAdd", 129: "FAdd", 130: "ISub", 131: "FSub", 132: "IMul", 133: "FMul", 134: "UDiv",
    135: "SDiv", 136: "FDiv", 137: "UMod", 138: "SRem", 139: "SMod", 140: "FRem", 141: "FMod",
    142: "VectorTimesScalar", 143: "MatrixTimesScalar", 144: "VectorTimesMatrix", 145: "MatrixTimesVector",
    146: "MatrixTimesMatrix", 148: "Dot", 149: "IAddCarry", 150: "ISubBorrow", 151: "UMulExtended",
    152: "SMulExtended", 154: "Any", 155: "All", 156: "IsNan", 157: "IsInf", 158: "IsFinite", 159: "IsNormal",
    160: "SignBitSet", 162: "Ordered", 163: "Unordered", 164: "LogicalEqual", 165: "LogicalNotEqual",
    166: "LogicalOr", 167: "LogicalAnd", 168: "LogicalNot", 169: "Select", 170: "IEqual", 171: "INotEqual",
    172: "UGreaterThan", 173: "SGreaterThan", 174: "UGreaterThanEqual", 175: "SGreaterThanEqual", 176: "ULessThan",
    177: "SLessThan", 178: "ULessThanEqual", 179: "SLessThanEqual", 180: "FOrdEqual", 181: "FUnordEqual",
    182: "FOrdNotEqual", 183: "FUnordNotEqual", 184: "FOrdLessThan", 185: "FUnordLessThan", 186: "FOrdGreaterThan",
    187: "FUnordGreaterThan", 188: "FOrdLessThanEqual", 189: "FUnordLessThanEqual", 190: "FOrdGreaterThanEqual",
    191: "FUnordGreaterThanEqual", 194: "ShiftRightLogical", 195: "ShiftRightArithmetic", 196: "ShiftLeftLogical",
    197: "BitwiseOr", 198: "BitwiseXor", 199: "BitwiseAnd", 200: "Not", 201: "BitFieldInsert",
    202: "BitFieldSExtract", 203: "BitFieldUExtract", 204: "BitReverse", 205: "BitCount", 207: "DPdx", 208: "DPdy",
    209: "Fwidth", 210: "DPdxFine", 211: "DPdyFine", 212: "FwidthFine", 213: "DPdxCoarse", 214: "DPdyCoarse",
    215: "FwidthCoarse", 218: "EmitVertex", 219: "EndPrimitive", 224: "ControlBarrier", 225: "MemoryBarrier",
    227: "AtomicLoad", 228: "AtomicStore", 229: "AtomicExchange", 230: "AtomicCompareExchange",
    232: "AtomicIIncrement", 233: "AtomicIDecrement", 234: "AtomicIAdd", 235: "AtomicISub", 236: "AtomicSMin",
    237: "AtomicUMin", 238: "AtomicSMax", 239: "AtomicUMax", 240: "AtomicAnd", 241: "AtomicOr", 242: "AtomicXor",
    245: "Phi", 246: "LoopMerge", 247: "SelectionMerge", 248: "Label", 249: "Branch", 250: "BranchConditional",
    251: "Switch", 252: "Kill", 253: "Return", 254: "ReturnValue", 255: "Unreachable", 330: "ModuleProcessed",
    331: "ExecutionModeId", 332: "DecorateId", 333: "GroupNonUniformElect", 334: "GroupNonUniformAll",
    335: "GroupNonUniformAny", 336: "GroupNonUniformAllEqual", 337: "GroupNonUniformBroadcast",
    338: "GroupNonUniformBroadcastFirst", 339: "GroupNonUniformBallot", 340: "GroupNonUniformInverseBallot",
    341: "GroupNonUniformBallotBitExtract", 342: "GroupNonUniformBallotBitCount", 343: "GroupNonUniformBallotFindLSB",
    344: "GroupNonUniformBallotFindMSB", 345: "GroupNonUniformShuffle", 346: "GroupNonUniformShuffleXor",
    347: "GroupNonUniformShuffleUp", 348: "GroupNonUniformShuffleDown", 349: "GroupNonUniformIAdd",
    366: "GroupNonUniformQuadSwap", 365: "GroupNonUniformQuadBroadcast", 400: "CopyLogical", 401: "PtrEqual",
    4416: "TerminateInvocation", 4421: "SubgroupBallotKHR", 4422: "SubgroupFirstInvocationKHR",
    4428: "SubgroupAllKHR", 4429: "SubgroupAnyKHR", 4430: "SubgroupAllEqualKHR", 4432: "SubgroupReadInvocationKHR",
    5294: "EmitMeshTasksEXT", 5295: "SetMeshOutputsEXT", 5364: "BeginInvocationInterlockEXT",
    5365: "EndInvocationInterlockEXT", 5380: "DemoteToHelperInvocation", 5381: "IsHelperInvocationEXT",
}
GLSL = {1: "Round", 2: "RoundEven", 3: "Trunc", 4: "FAbs", 5: "SAbs", 6: "FSign", 7: "SSign", 8: "Floor", 9: "Ceil",
        10: "Fract", 13: "Sin", 14: "Cos", 15: "Tan", 18: "Atan", 25: "Atan2", 26: "Pow", 27: "Exp", 28: "Log",
        29: "Exp2", 30: "Log2", 31: "Sqrt", 32: "InverseSqrt", 37: "FMin", 38: "UMin", 39: "SMin", 40: "FMax",
        41: "UMax", 42: "SMax", 43: "FClamp", 44: "UClamp", 45: "SClamp", 46: "FMix", 50: "Fma", 53: "Ldexp",
        54: "PackSnorm4x8", 55: "PackUnorm4x8", 56: "PackSnorm2x16", 57: "PackUnorm2x16", 58: "PackHalf2x16",
        59: "PackDouble2x32", 60: "UnpackSnorm2x16", 61: "UnpackUnorm2x16", 62: "UnpackHalf2x16",
        63: "UnpackSnorm4x8", 64: "UnpackUnorm4x8", 65: "UnpackDouble2x32", 66: "Length", 69: "Normalize",
        73: "FindILsb", 74: "FindSMsb", 75: "FindUMsb", 79: "NMin", 80: "NMax", 81: "NClamp"}
SC = {0: "UniformConstant", 1: "Input", 2: "Uniform", 3: "Output", 4: "Workgroup", 6: "Private", 7: "Function",
      9: "PushConstant", 11: "Image", 12: "StorageBuffer", 5349: "PhysicalStorageBuffer"}
DEC = {2: "Block", 3: "BufferBlock", 6: "ArrayStride", 11: "BuiltIn", 13: "NoPerspective", 14: "Flat",
       18: "Invariant", 19: "Restrict", 20: "Aliased", 21: "Volatile", 23: "Coherent", 24: "NonWritable",
       25: "NonReadable", 30: "Location", 31: "Component", 32: "Index", 33: "Binding", 34: "DescriptorSet",
       35: "Offset", 5300: "NonUniform"}
# opcodes with a result type and a result id (the rest of the core ones above have neither, or only a result id)
NO_RESULT = {3, 4, 5, 6, 8, 10, 14, 15, 16, 17, 56, 62, 63, 71, 72, 218, 219, 224, 225, 228, 246, 247, 249, 250,
             251, 252, 253, 254, 255, 330, 331, 332, 4416, 5294, 5295, 5364, 5365, 5380}
RESULT_ONLY = {11, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 39, 248}


def read(path):
    data = open(path, "rb").read()
    w = struct.unpack("<%dI" % (len(data) // 4), data[:len(data) // 4 * 4])
    if not w or w[0] != 0x07230203:
        sys.exit(f"{path}: not SPIR-V")
    return w


def insts(w):
    i = 5
    while i < len(w):
        op, n = w[i] & 0xFFFF, w[i] >> 16
        if n == 0 or i + n > len(w):
            break
        yield i, op, w[i + 1:i + n]
        i += n


def lit_string(ops):
    b = b"".join(struct.pack("<I", x) for x in ops)
    return b.split(b"\0", 1)[0].decode("utf-8", "replace"), (len(b.split(b"\0", 1)[0]) + 4) // 4


class Module:
    def __init__(self, path):
        self.w = read(path)
        self.bound = self.w[3]
        self.names, self.types, self.consts, self.ext_glsl = {}, {}, {}, None
        self.funcs = []          # (id, start index, [instructions])
        self.decor = defaultdict(list)
        cur = None
        for i, op, a in insts(self.w):
            if op == 5:
                self.names[a[0]] = lit_string(a[1:])[0]
            elif op == 11 and lit_string(a[1:])[0] == "GLSL.std.450":
                self.ext_glsl = a[0]
            elif op == 71:
                self.decor[a[0]].append(tuple(a[1:]))
            elif op in RESULT_ONLY and op != 248:
                self.types[a[0]] = (op, a[1:])
            elif op == 43:
                self.consts[a[1]] = a[2:]
            if op == 54:
                cur = [a[1], i, []]
                self.funcs.append(cur)
            if cur is not None:
                cur[2].append((i, op, a))
            if op == 56:
                cur = None

    def nm(self, x):
        return f"%{self.names[x]}_{x}" if x in self.names else f"%{x}"

    def type_str(self, t, depth=0):
        if t not in self.types or depth > 4:
            return self.nm(t)
        op, a = self.types[t]
        if op == 19: return "void"
        if op == 20: return "bool"
        if op == 21: return ("i" if a[1] else "u") + str(a[0])
        if op == 22: return "f" + str(a[0])
        if op == 23: return f"vec{a[1]}<{self.type_str(a[0], depth + 1)}>"
        if op == 28: return f"{self.type_str(a[0], depth + 1)}[{self.consts.get(a[1], ['?'])[0]}]"
        if op == 29: return f"{self.type_str(a[0], depth + 1)}[]"
        if op == 30: return "struct{" + ",".join(self.type_str(x, depth + 1) for x in a) + "}"
        if op == 32: return f"ptr<{SC.get(a[0], a[0])},{self.type_str(a[1], depth + 1)}>"
        return f"{OPS.get(op, op)}{self.nm(t)}"


def disasm(path, out):
    m = Module(path)
    f = open(out, "w", encoding="utf-8") if out else sys.stdout
    for i, op, a in insts(m.w):
        name = OPS.get(op, f"Op{op}")
        if op in (5, 6, 7, 10, 330):
            f.write(f"Op{name} {' '.join(map(str, a[:1 if op != 6 else 2]))} \"{lit_string(a[1 if op != 6 else 2:])[0]}\"\n")
            continue
        if op == 17:
            f.write(f"OpCapability {a[0]}\n")
            continue
        if op == 71:
            f.write(f"OpDecorate {m.nm(a[0])} {DEC.get(a[1], a[1])} {' '.join(map(str, a[2:]))}\n")
            continue
        if op == 72:
            f.write(f"OpMemberDecorate {m.nm(a[0])} {a[1]} {DEC.get(a[2], a[2])} {' '.join(map(str, a[3:]))}\n")
            continue
        if op in RESULT_ONLY:
            res, rest = a[0], a[1:]
            if op == 248:
                f.write(f"{m.nm(res)} = OpLabel\n")
            else:
                f.write(f"{m.nm(res)} = Op{name} {m.type_str(res)}\n")
            continue
        if op in NO_RESULT:
            if op == 251:
                cases = " ".join(f"{a[k]}:{m.nm(a[k + 1])}" for k in range(2, len(a) - 1, 2))
                f.write(f"      OpSwitch {m.nm(a[0])} default:{m.nm(a[1])} {cases}\n")
            else:
                f.write(f"      Op{name} {' '.join(m.nm(x) if op not in (15, 16, 14) else str(x) for x in a)}\n")
            continue
        if len(a) < 2:
            f.write(f"      Op{name} {' '.join(map(str, a))}\n")
            continue
        rt, res, rest = a[0], a[1], a[2:]
        if op == 43:
            body = " ".join(str(x) for x in rest)
        elif op == 59:
            body = f"{SC.get(rest[0], rest[0])}" + (f" {m.nm(rest[1])}" if len(rest) > 1 else "")
        elif op == 12 and rest and rest[0] == m.ext_glsl:
            body = f"GLSL.{GLSL.get(rest[1], rest[1])} " + " ".join(m.nm(x) for x in rest[2:])
        elif op in (81, 82):   # composite extract/insert: literal indices after the ids
            nid = 1 if op == 81 else 2
            body = " ".join(m.nm(x) for x in rest[:nid]) + " " + " ".join(str(x) for x in rest[nid:])
        elif op == 79:
            body = " ".join(m.nm(x) for x in rest[:2]) + " " + " ".join(str(x) for x in rest[2:])
        else:
            body = " ".join(m.nm(x) for x in rest)
        f.write(f"{m.nm(res)} = Op{name} {m.type_str(rt)} {body}\n")
    if out:
        f.close()


def func_stats(m, fn):
    fid, start, body = fn
    c = Counter(op for _, op, _ in body)
    calls = [a[2] for _, op, a in body if op == 57]
    return {"id": fid, "name": m.names.get(fid, ""), "words": sum(1 + len(a) for _, _, a in body),
            "insts": len(body), "blocks": c[248], "loops": c[246], "selections": c[247], "switches": c[251],
            "phis": c[245], "calls": calls, "loads": c[61], "stores": c[62], "access_chains": c[65] + c[66],
            "vars": c[59], "ops": c}


def stats(paths):
    for p in paths:
        m = Module(p)
        allc = Counter(op for _, op, _ in insts(m.w))
        caps = [a[0] for _, op, a in insts(m.w) if op == 17]
        exts = [lit_string(a)[0] for _, op, a in insts(m.w) if op == 10]
        print(f"== {p}: {len(m.w)} words, bound {m.bound}, {len(m.funcs)} functions")
        print("   capabilities", caps, "extensions", exts)
        sc = Counter()
        for _, op, a in insts(m.w):
            if op == 59 and len(a) >= 3:
                sc[SC.get(a[2], a[2])] += 1
        print("   variables by storage class", dict(sc))
        blocks = Counter()
        for t, d in m.decor.items():
            for x in d:
                if x[0] in (2, 3):
                    blocks[DEC[x[0]]] += 1
        print("   block decorations", dict(blocks))
        print("   top opcodes", ", ".join(f"{OPS.get(o, o)} {n}" for o, n in allc.most_common(25)))
        for fn in m.funcs:
            s = func_stats(m, fn)
            print(f"   fn {m.nm(s['id'])}: {s['words']} words, {s['insts']} insts, {s['blocks']} blocks, {s['loops']} loops, "
                  f"{s['selections']} sel, {s['switches']} sw, {s['phis']} phi, {len(s['calls'])} calls, "
                  f"{s['loads']} loads, {s['stores']} stores, {s['access_chains']} access chains, {s['vars']} vars")


def funcs(path):
    m = Module(path)
    callers = defaultdict(Counter)
    for fn in m.funcs:
        for _, op, a in fn[2]:
            if op == 57:
                callers[a[2]][fn[0]] += 1
    for fn in m.funcs:
        s = func_stats(m, fn)
        params = [a for _, op, a in fn[2] if op == 55]
        print(f"{m.nm(s['id'])}: {s['words']} words, {s['blocks']} blocks, params {len(params)} "
              f"({', '.join(m.type_str(a[0]) for a in params)}), calls {Counter(map(m.nm, s['calls'])).most_common(8)}, "
              f"called by {[(m.nm(k), v) for k, v in callers[s['id']].items()]}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "dis":
        out = args[args.index("-o") + 1] if "-o" in args else None
        disasm(args[0], out)
    elif cmd == "stats":
        stats(args)
    elif cmd == "funcs":
        funcs(args[0])
    else:
        sys.exit(__doc__)
