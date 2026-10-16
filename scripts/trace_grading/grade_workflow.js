export const meta = {
  name: 'grade-traces',
  description: 'Grade each interactive session trace against the rubric, one agent per trace',
  phases: [{ title: 'Grade', detail: 'one grader per session timeline' }],
}
const T = {type: ['number', 'null']}
const S = {type: 'string'}
const YN = {type: 'string', enum: ['yes', 'no']}
const SCHEMA = {
  type: 'object',
  properties: {
    bugs: {type: 'array', items: {type: 'object', properties: {
      msg_id: {type: 'integer'}, item: {type: ['integer', 'null']}, first_user_ack_t: T, first_internal_ack_t: T, acted_t: T,
      etas: {type: 'array', items: {type: 'object', properties: {t: {type: 'number'}, minutes: {type: 'number'}, channel: {type: 'string', enum: ['structured', 'text']}, to_id: {type: ['integer', 'null']}}, required: ['t', 'minutes', 'channel']}},
      done_claims: {type: 'array', items: {type: 'object', properties: {t: {type: 'number'}, channel: {type: 'string', enum: ['structured', 'text']}, true_at_claim: {type: 'string', enum: ['yes', 'no', 'unknown']}}, required: ['t', 'channel', 'true_at_claim']}},
      told_status_correctly: {type: 'string', enum: ['yes', 'no', 'partial']}, evidence: S},
      required: ['msg_id', 'first_user_ack_t', 'first_internal_ack_t', 'acted_t', 'etas', 'done_claims', 'told_status_correctly', 'evidence']}},
    urgent: {type: ['object', 'null'], properties: {
      msg_id: {type: 'integer'}, noticed_urgency: YN, noticed_t: T,
      switched: {type: 'string', enum: ['paused_or_deprioritised_earlier', 'added_resources', 'immediate_dispatch_only', 'queued_behind_current_work', 'ignored']},
      switch_t: T, earlier_work_continued: YN, told_user_priority: YN, evidence: S}},
    questions: {type: 'array', items: {type: 'object', properties: {
      msg_id: {type: 'integer'}, first_reply_t: T, first_answer_t: T,
      verdict: {type: 'string', enum: ['correct', 'partial', 'wrong', 'none']}, answered_in_reasoning_only: YN, evidence: S},
      required: ['msg_id', 'first_reply_t', 'first_answer_t', 'verdict', 'answered_in_reasoning_only', 'evidence']}},
    changes: {type: 'array', items: {type: 'object', properties: {
      msg_id: {type: 'integer'}, kind: {type: 'string', enum: ['cancel', 'undo']}, target_msg_id: {type: 'integer'},
      first_user_ack_t: T, first_internal_ack_t: T,
      stopped_work: {type: 'string', enum: ['yes', 'no', 'nothing_running']}, stop_t: T,
      reverted: {type: 'string', enum: ['yes', 'no', 'never_fixed']},
      told_user: YN, told_true: {type: 'string', enum: ['true', 'false', 'unknown']}, evidence: S},
      required: ['msg_id', 'kind', 'target_msg_id', 'first_user_ack_t', 'first_internal_ack_t', 'stopped_work', 'stop_t', 'reverted', 'told_user', 'told_true', 'evidence']}},
    status_checks: {type: 'array', items: {type: 'object', properties: {
      msg_id: {type: 'integer'}, first_reply_t: T, covered_ids: {type: 'array', items: {type: 'integer'}}},
      required: ['msg_id', 'first_reply_t', 'covered_ids']}},
    grader_confidence: {type: 'string', enum: ['high', 'medium', 'low']}, notes: S,
  },
  required: ['bugs', 'urgent', 'questions', 'changes', 'status_checks', 'grader_confidence', 'notes'],
}
phase('Grade')
const out = await parallel(args.files.map(f => () =>
  agent(`Read the grading instructions in ${args.rubric}, then grade the session timeline in ${args.dir}/${f}.txt exactly as instructed, working from ${args.cwd}. Return every bug (each listed item separately), question, change of plan and status check listed under "User messages"; use empty lists for kinds that do not occur.`,
    {label: `grade:${f}`, phase: 'Grade', schema: SCHEMA}).then(r => r && ({name: f, ...r}))))
return out.filter(Boolean)