# Pseudocode conventions

The algorithms use the following language-neutral notation.

| Form | Meaning |
| --- | --- |
| `MODULE / PURPOSE / PAPER` | Module identity, responsibility and source equations |
| `CLASS / PROCEDURE` | A declared object or algorithm |
| `INPUT / PARAMETERS / OUTPUT` | Data, configuration and returned values |
| `x <- expression` | Assignment; `=` in a condition means equality |
| `A[B,H,N,K]` | Tensor with named axes; shapes may omit a singleton batch axis |
| `record.field`, `mapping[key]` | Named record field and dictionary lookup |
| `FOR / IF / WHILE / END` | Control flow, indented by four spaces |
| `RETURN` | Result of the procedure |
| `REQUIRE / ASSERT` | Input condition and invariant checked by the procedure |
| `Pending(missing)` | Required configuration or artifact names are returned to the caller |
| `Failure(reason,...)` | An incompatible input stops the current operation |
| `RAISE Error(reason,...)` | Propagate an invalid-input or invariant failure to the caller |
| `Undefined(reason)` | A metric has no value for the selected population |
| `Empty(reason)` | The requested selection has no records |

Array indices begin at zero. Retained destination slots are `0..R-1`; the residual slot is `R`, with `K=R+1`. A range such as `0..H-1` includes both endpoints. Timestamps keep their time zone; interval notation `[start,end)` excludes the endpoint.

Named tuple keys expose their fields, such as `key.section`. Dictionary iteration uses `KEYS`, `VALUES`, or `(key,value)`. Tensor reductions name the axis when it affects the result.

Standard mathematical operations and collection/file operations appear in capitals. A configurable operation such as `kernel.delay(hops)` is supplied through a named parameter contract. Algorithm-specific procedures are declared in the module or called with their module name. Parameter and transform IDs identify the configuration used across modules.
