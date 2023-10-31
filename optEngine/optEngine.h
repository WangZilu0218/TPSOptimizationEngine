#ifndef OPTENGINE_H
#define OPTENGINE_H

#include "common.cuh"
#include "csc.h"
#include "fista.h"
enum {FISTA, IPOPT};
class optEngine()
/*

*/
{
protected:

public:
    // variable
    CSC dosemap;
    std::string method;

    // function
    optEngine();
    ~optEngine();
    void optimize(); // initialize all arrays, call correct optimize method
    void FISTAOptmize();
    void IPOPT();

}

#endif