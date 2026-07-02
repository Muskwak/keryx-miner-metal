// Bridging header — Rust C FFI declarations
// These are also declared in Swift directly via @_silgen_name,
// but Xcode's build system may need the Obj-C header for linking.

#ifndef Bridge_h
#define Bridge_h

#include <stdbool.h>

extern bool keryx_miner_set_doc_path(const char *path);
extern bool keryx_miner_initialize(void);
extern bool keryx_miner_connect(const char *address);
extern bool keryx_miner_set_mining_address(const char *address);
extern bool keryx_miner_start(void);
extern void keryx_miner_stop(void);
extern char *keryx_miner_status(void);
extern void keryx_miner_free_string(char *s);

#endif /* Bridge_h */
