global false = 0;
global true = 1;  // temporary
/*intrinsic*/def len(/*list*/val a) {}
/*intrinsic*/def delitem(/*list*/val a, number idx) {}

global[] psHeap = [];
global[] _psHeap_freelist = [];

def _PsHeap_FinalizeObj(number ptr) {
    // Dispatch finalizer here, decref components, etc.
}

global __result__;  // TODO: some syntax sugar for return?
def PsHeap_AllocWord() {
    if len(_psHeap_freelist) > 0 {
        __result__ = _psHeap_freelist[0];
        delitem(_psHeap_freelist, 0);
        if COUNT_REFS {
            refStack[__result__] = 1;
        }
    } else {
        // TODO: figure out list syntax
        psHeap.append('');
        __result__ = len(psHeap);
        if COUNT_REFS {
            refStack.append(1);
        }
    }
}
/** Note: Doesn't finalize! */
def PsHeap_FreeWord(number ptr) {
    psHeap[ptr] = '';  // Next person can't read what was here
    if COUNT_REFS {
        refStack[ptr] = 0;
    }
    _psHeap_freelist.append(ptr);
}
def PsHeap_DeleteWord(number ptr) {
    _PsHeap_FinalizeObj(ptr);
    PsHeap_FreeWord(ptr);
}

global[] refStack = [];
def decref(number ptr) {
    if refStack[ptr] >= 0 {  // negative refcount indicates immortal objects
        refStack[ptr] -= 1;
        if refStack[ptr] == 0 {
            PsHeap_DeleteWord(ptr);
        }
    }
}
def incref(number ptr) {
    if refStack[ptr] >= 0 {  // negative refcount indicates immortal objects
        refStack[ptr] += 1;
    }
}
// implement GC that can do cycles


global[] stack = [];
global COUNT_REFS = false;

// opcodes from python version 3.8:

// opcodes from: https://unpyc.sourceforge.net/Opcodes.html
// and https://harrisonmorgan.dev/2020/04/13/learning-cpython-bytecode-instructions/
// also https://docs.python.org/3.8/library/dis.html#opcode-NOP     for description
// and https://github.com/python/cpython/blob/3.8/Lib/opcode.py#L58 for numbers

def NOP() {}

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
    let tos1 = stack[tos_idx - 1];
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
    let tos0 = stack[tos_idx];
    stack[tos_idx] = stack[tos_idx - 1];
    stack[tos_idx - 1] = stack[tos_idx - 2];
    stack[tos_idx - 2] = stack[tos_idx - 3];
    stack[tos_idx - 3] = tos0;
}
