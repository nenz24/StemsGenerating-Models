# Raw ML Program candidate rejected by CoreAI

The user's device report records `Source asset validation: false`, followed by
`AIModelCacheError.failedToSpecialize` (domain CoreAIDelegates.AIModelCacheError,
code 3): the `affine.mlpackage` asset is malformed, `Missing hash file`.
Both model-loaded and model-inference flags are NO. Structured evidence is in
[report.json](report.json); the transient app-container UUID is not retained.

This rejects the raw source package on this tested loader path. It does not
establish operator incompatibility, bad numerical parity, insufficient memory
or a device/accelerator failure: inference never began. Do not create an ad hoc
hash file to satisfy the diagnostic or claim to know the native hash schema.

The user corrected the expected native format to `.aimodels` and supplied
https://developer.apple.com/core-ai/ . The proposed `.mlmodelc` follow-up was
cancelled before publication or execution. The next step requires the documented
CoreAI-native exporter and asset, not an invented hash or renamed legacy package.
The official page could not be retrieved from this environment (proxy HTTP 403);
no export commands or converter names are inferred from its unread contents.
