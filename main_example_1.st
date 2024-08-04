global stack = list();
global COUNT_REFS = false;

// opcodes from python version 3.8:

// opcodes from: https://unpyc.sourceforge.net/Opcodes.html
// and https://harrisonmorgan.dev/2020/04/13/learning-cpython-bytecode-instructions/
// also https://docs.python.org/3.8/library/dis.html#opcode-NOP     for description
// and https://github.com/python/cpython/blob/3.8/Lib/opcode.py#L58 for numbers

def POP_TOP() {
    if COUNT_REFS {
        decref(stack[len(stack)]);
    }
    delitem(stack, len(stack));
}

def ROT_TWO() {
    let tos_idx = len(stack);
    let tos0 = stack[tos_idx];
    stack[tos_idx] = stack[tos_idx - 1];
    stack[tos_idx - 1] = tos0;
}

def ROT_THREE() {
    // top   = top-1
    // top-1 = top-2
    // top-2 = top
    // i.e. move 2nd and 3rd up and move top to 3rd
    let tos_idx = len(stack);
    let tos0 = stack[tos_idx];
    stack[tos_idx] = stack[tos_idx - 1];
    stack[tos_idx - 1] = stack[tos_idx - 2];
    stack[tos_idx - 2] = tos0;
}

def DUP_TOP() {
    let value = stack[len(stack)];
    stack.append(value);
    if COUNT_REFS {
        incref(value);
    }
}

def DUP_TOP_TWO() {
    let tos_idx = len(stack);
    let tos0 = stack[tos_idx];
    let tos1 = stack[tod_idx - 1];
    stack.append(tos1);
    stack.append(tos0);  // keep tos0 at top
    if COUNT_REFS {
        incref(tos0);
        incref(tos1);
    }
}

def ROT_FOUR() {
    // tos3 = tos
    // tos = tos1
    // tos1 = tos2
    // tos2 = tos3
    let tos_idx = len(stack);
    let tos0 = stack[tos];
    stack[tos_idx] = stack[tos_idx - 1];
    stack[tos_idx - 1] = stack[tos_idx - 2];
    stack[tos_idx - 2] = stack[tos_idx - 3];
    stack[tos_idx - 3] = tos0;
}

def NOP() {}


