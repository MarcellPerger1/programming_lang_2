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
    _psHeap_freelist__len();
    if __result__ > 0 {
        __result__ = _psHeap_freelist[0];
        _psHeap_freelist__delitem(0);
        if COUNT_REFS != 0 {
            refStack[__result__] = 1;
        }
    } else {
        // TODO: figure out list syntax
        psHeap__append('');
        psHeap__len();  // sets __result__
        if COUNT_REFS != 0 {
            refStack__append(1);
        }
    }
}
/** Note: Doesn't finalize! */
def PsHeap_FreeWord(number ptr) {
    psHeap[ptr] = '';  // Next person can't read what was here
    if COUNT_REFS != 0 {
        refStack[ptr] = 0;
    }
    _psHeap_freelist__append(ptr);
}
def PsHeap_DeleteWord(number ptr) {
    _PsHeap_FinalizeObj(ptr);
    PsHeap_FreeWord(ptr);
}

global[] refStack = [];
/*intrinsic method*/def refStack__append(val x) {}
/*intrinsic method*/def stack__append(val x) {}
/*intrinsic method*/def lenStack() {}
/*intrinsic method*/def stackDelitem(val x) {}
/*intrinsic method*/def psHeap__append(val x) {}
/*intrinsic method*/def _psHeap_freelist__append(val x) {}
/*intrinsic method*/def _psHeap_freelist__delitem(val x) {}
/*intrinsic method*/def _psHeap_freelist__len() {}
/*intrinsic method*/def psHeap__len() {}

def decref(number ptr) {
    if refStack[ptr] >= 0 {  // negative refcount indicates immortal objects
        refStack[ptr] = refStack[ptr] - 1;
        if refStack[ptr] == 0 {
            PsHeap_DeleteWord(ptr);
        }
    }
}
def incref(number ptr) {
    if refStack[ptr] >= 0 {  // negative refcount indicates immortal objects
        refStack[ptr] = refStack[ptr] + 1;  // TODO: support += on list accesses
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
    lenStack();  // -> __result__
    if COUNT_REFS == true {
        decref(stack[__result__]);
    }
    stackDelitem(__result__);
}

def ROT_TWO() {
    lenStack();
    let tos_idx = __result__;
    let tos0 = stack[tos_idx];
    stack[tos_idx] = stack[tos_idx - 1];
    stack[tos_idx - 1] = tos0;
}

def ROT_THREE() {
    // top   = top-1
    // top-1 = top-2
    // top-2 = top
    // i.e. move 2nd and 3rd up and move top to 3rd
    lenStack();
    let tos_idx = __result__;
    let tos0 = stack[tos_idx];
    stack[tos_idx] = stack[tos_idx - 1];
    stack[tos_idx - 1] = stack[tos_idx - 2];
    stack[tos_idx - 2] = tos0;
}

def DUP_TOP() {
    lenStack();
    let value = stack[__result__];
    stack__append(value);
    if COUNT_REFS != false {
        incref(value);
    }
}

def DUP_TOP_TWO() {
    lenStack();
    let tos_idx = __result__;
    let tos0 = stack[tos_idx];
    let tos1 = stack[tos_idx - 1];
    stack__append(tos1);
    stack__append(tos0);  // keep tos0 at top
    if COUNT_REFS != 0 {
        incref(tos0);
        incref(tos1);
    }
}

def ROT_FOUR() {
    // tos3 = tos
    // tos = tos1
    // tos1 = tos2
    // tos2 = tos3
    lenStack();
    let tos_idx = __result__;
    let tos0 = stack[tos_idx];
    stack[tos_idx] = stack[tos_idx - 1];
    stack[tos_idx - 1] = stack[tos_idx - 2];
    stack[tos_idx - 2] = stack[tos_idx - 3];
    stack[tos_idx - 3] = tos0;
}
