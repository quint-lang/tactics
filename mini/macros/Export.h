//
// Created by kingkiller on 2023/10/14.
//

#ifndef TACTICS_MINI_MACROS_EXPORT_H
#define TACTICS_MINI_MACROS_EXPORT_H

#ifndef MINI_USING_CUSTOM_GENRATED_MACROS
#include <mini/macros/cmake_macros.h>
#endif // MINI_USING_CUSTOM_GENRATED_MACROS

#ifdef _WIN32
#define MINI_HIDDEN
#if defined(MINI_BUILD_SHARED_LIBS)
#define MINI_EXPORT __declspec(dllexport)
#define MINI_IMPORT __declspec(dllimport)
#else
#define MINI_EXPORT
#define MINI_IMPORT
#endif
#else
#if defined(__GNUC__)
#define MINI_EXPORT __attribute__((__visibility__("default")))
#define MINI_HIDDEN __attribute__((__visibility__("hidden")))
#else // defined(__GNUC__)
#define C10_EXPORT
#define MINI_HIDDEN
#endif // defined(__GNUC__)
#define MINI_IMPORT MINI_EXPORT
#endif

#ifdef NO_EXPORT
#undef MINI_EXPORT
#define MINU_EXPORT
#endif

// This one is being used by libmini.so
#ifdef MINI_BUILD_MAIN_LIB
#define MINI_API MINI_EXPORT
#else
#define MINI_API MINI_IMPORT
#endif

#endif //TACTICS_MINI_MACROS_EXPORT_H
