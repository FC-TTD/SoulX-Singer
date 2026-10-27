# Hub provider build and admission

User authorized this provider adoption and machine credentials on2026-10-27. Build component explicitly through ./deploy.sh build --component <name> --work-dir <evidence> --launcher <registered TTD scripts/ttd-ansible-playbook>. Source is a committed read-only archive; native base and shared SDK0.3.0 artifacts are pinned. Thin COPY--link overlay has no package install or base unpack step. Shared cache family: registry.ttd/music-native/shared-cache:buildcache. Never rebuild Core/node/SDK implicitly.

Before first registration, prepare stopped candidates with fresh actor/private token and persistent journal; preserve native weights/precision/queue. Close ordinary admission, drain the old independent provider and keep native input/output state. Only real scoped native output, activity/queue completion, process-group exit and NVML release evidence allow opening. Unknown work stays fenced and is never automatically killed/replayed. CPU fixtures do not prove GPU admission or native model acceptance. The source stamp and immutable image are deployment identities; current state lives in the release receipts, not this file.

Initial candidates use Edge GPU1 through Hub; existing independent song-audio providers still occupy GPU0 during preparation. This preserves their availability during native acceptance. Capacity pressure may reclaim only actually idle managed models through the existing Hub protocol. Physical placement changes to Edge GPU1 retain native cuda:0 inside the assigned interpreter.

During candidate validation, shared historical task state remains intact. Recovery is scoped to the current actor; legacy jobs are not reclassified or replayed by candidate startup. Ordinary traffic stays on the original provider until native acceptance passes and the original ingress is fenced/drained.

原生生命周期：这些接口原来即按任务启动CLI、加载模型并在CLI结束后释放权重。适配保留该行为；SDK引擎负责分配UUID、准入、执行和真实进程组完成屏障，保留热引擎，不添加独立TTL或改变模型驻留方式。在途/原生等待/未知执行保护租约；空闲引擎只由Hub压力回收或显式排空。

构建禁止新增RUN/WORKDIR文件系统步骤，避免为了薄适配解包多GB基础层。模型基础镜像和SDK通过只读COPY--link叠加，保留原CWD和环境。Native CLI不接收Hub机器凭据。
