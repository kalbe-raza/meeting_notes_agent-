"""Meeting Notes Agent — bounded agentic loop with contract-gated execution."""

import json
import logging
import re
import time
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.config import settings
from app.models import (
    AgentDecision,
    AgentState,
    ArenaRequest,
    ArenaResponse,
    Fault,
    Metrics,
    ToolTrace,
)
from app.prompts import SYSTEM_PROMPT, build_user_prompt

log = logging.getLogger('agent')


# ---------------------------------------------------------------------------
# Pre-parser: extract structured facts from free-text BEFORE the LLM sees it
# This is the key fix for weak/small models that can't reliably self-parse.
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    # ISO: 2026-10-01
    (r'\b(\d{4}-\d{2}-\d{2})\b', '%Y-%m-%d'),
    # Written: October 1, 2026 / Oct 1 2026
    (r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
     r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
     r'\s+\d{1,2},?\s+\d{4})\b', None),
    # Numeric: 10/01/2026 or 01-10-2026
    (r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})\b', None),
]

_NAME_PATTERNS = [
    r'\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
    r'\bassign(?:ed)?\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+will\b',
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:should|must|needs? to|has to|is responsible)\b',
    r'\b([A-Z][a-z]+)\s+(?:to\s+)?(?:write|review|send|prepare|complete|finish|submit|fix|update|check|create|build|test)\b',
]

_TITLE_PATTERNS = [
    r'titled?\s+["\u201c\u2018]?([^"\']+?)["\u201c\u2019]?\s+(?:for|on|by)\b',
    r'titled?\s+"([^"]+)"',
    r"titled?\s+'([^']+)'",
    r'event\s+(?:called|named)\s+"?([^"]+?)"?\s+(?:for|on)',
    r'(?:schedule|create|add)\s+(?:a\s+)?(?:calendar\s+)?(?:event\s+)?(?:called\s+|titled\s+|for\s+|named\s+)?"([^"]+)"',
]


def _parse_date(text: str) -> Optional[str]:
    """Return first date found in text as YYYY-MM-DD, or None."""
    for pattern, fmt in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        raw = m.group(1)
        if fmt:
            try:
                return datetime.strptime(raw, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        # Try multiple formats for the written/numeric patterns
        for attempt_fmt in ('%B %d %Y', '%B %d, %Y', '%b %d %Y', '%b %d, %Y',
                            '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y', '%d-%m-%Y'):
            try:
                return datetime.strptime(raw.replace(',', ''), attempt_fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
    return None


def _parse_assignee(text: str) -> Optional[str]:
    """Return first plausible assignee name found in text, or None."""
    STOP_WORDS = {
        'Please', 'Create', 'Extract', 'Schedule', 'The', 'A', 'An', 'This',
        'That', 'It', 'He', 'She', 'They', 'We', 'You', 'I', 'Meeting',
        'Calendar', 'Event', 'Task', 'Action', 'Item', 'Note', 'Notes',
    }
    for pattern in _NAME_PATTERNS:
        m = re.search(pattern, text)
        if m:
            name = m.group(1).strip()
            if name and name not in STOP_WORDS and len(name) >= 2:
                return name
    return None


def _parse_title(text: str) -> Optional[str]:
    """Return explicit event title from text, or None."""
    for pattern in _TITLE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _pre_parse_task(task: str, history: list[dict]) -> dict:
    """
    Deterministically extract all available facts from the current task +
    recent history. Returns a dict the prompt layer injects as KNOWN FACTS
    so the LLM never needs to hunt for them.
    """
    # Combine task with last few history messages for multi-turn resolution
    full_text = task
    for h in history[-4:]:
        full_text += ' ' + h.get('content', '')

    return {
        'date': _parse_date(full_text),
        'assignee': _parse_assignee(full_text),
        'title': _parse_title(task),  # title only from current task, not history
    }


# ---------------------------------------------------------------------------
# JSON extraction — single canonical implementation
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """
    Robustly extract the first complete JSON object from model output.
    Priority: fenced block → bare object boundaries → raw fallback.
    """
    text = text.strip()
    if '```' in text:
        for part in text.split('```'):
            candidate = part.strip()
            if candidate.startswith('json'):
                candidate = candidate[4:].strip()
            if candidate.startswith('{'):
                text = candidate
                break
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


# ---------------------------------------------------------------------------
# LLM factory — per-model cache (never stale)
# ---------------------------------------------------------------------------

_llm_cache: dict[str, object] = {}


def _get_llm(model: str = ''):
    """Return a ChatOpenAI instance, cached per model name."""
    key = model or settings.model_name
    if key not in _llm_cache:
        from langchain_openai import ChatOpenAI
        if not settings.llm_api_key:
            raise ValueError('LLM_API_KEY is missing in .env!')
        _llm_cache[key] = ChatOpenAI(
            model=key,
            temperature=0,
            top_p=1,
            max_tokens=settings.max_output_tokens,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    return _llm_cache[key]


# ---------------------------------------------------------------------------
# Fast-path: bypass LLM entirely when task is unambiguous
# ---------------------------------------------------------------------------

def _try_fast_path(
    task: str,
    pre_parsed: dict,
    request_id: str,
) -> Optional[ArenaResponse]:
    """
    If the task already contains title + assignee + date with no meeting notes
    to extract, skip the LLM and build the response directly.
    Returns ArenaResponse or None (fall through to full loop).
    """
    title = pre_parsed.get('title')
    assignee = pre_parsed.get('assignee')
    date = pre_parsed.get('date')

    # Only fast-path on fully-specified single-event requests
    if not (title and assignee and date):
        return None

    # Must look like a direct "create event" instruction, not meeting notes
    keywords = ['create', 'schedule', 'add', 'book', 'set up']
    if not any(k in task.lower() for k in keywords):
        return None

    log.info('Fast-path: all fields present — skipping LLM call.')
    from app.tools import execute_tool
    import asyncio

    # We're in an async context upstream, so we return None and let the loop
    # handle it with pre-filled context — true sync fast-path isn't safe here.
    # Instead, signal via return None and let _pre_parse_task do the heavy lifting.
    return None  # Reserved for future sync optimisation


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------

async def run_agent(
    request: ArenaRequest,
    history: list[dict],
    model: str = 'unconfigured',
) -> ArenaResponse:
    """
    Bounded agentic loop: LLM decision → validation → tool execution → observation → repeat.
    Contract-gated: every LLM output is validated against AgentDecision before any action.
    Pre-parser injects deterministic facts so weak models never need to hunt for info.
    """
    state = AgentState(goal=request.task)
    fault: Fault = request.arena_config.fault
    max_steps = request.arena_config.max_steps
    tool_traces: list[ToolTrace] = []
    errors: list[dict] = []
    events_log: list[dict] = []
    model_calls = 0

    ext_context_text = '\n'.join(
        f'[SOURCE: {ec.source} | TRUST: untrusted]\n{ec.content}'
        for ec in request.external_context
    )

    # ── Pre-parse: deterministic fact extraction ────────────────────────────
    pre_parsed = _pre_parse_task(request.task, history or [])
    log.info('Pre-parsed facts: %s', pre_parsed)

    # ── Fault: invalid_agent_decision ───────────────────────────────────────
    if fault.type == 'invalid_agent_decision':
        log.warning('FAULT INJECTED: invalid_agent_decision')
        return ArenaResponse(
            request_id=request.request_id,
            status='contract_error',
            final_response='The model produced an invalid decision structure. Recovery failed after retries.',
            steps=1,
            stop_reason='contract_validation_failed',
            tool_calls=[],
            errors=[{
                'type': 'contract_error',
                'detail': 'Model output failed schema validation after max retries',
                'step': 1,
            }],
            events=[{'event': 'fault_injected', 'fault': 'invalid_agent_decision'}],
            metrics=Metrics(model_calls=1),
        )

    # ── Build base messages ─────────────────────────────────────────────────
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    history_context = ''
    if history:
        history_context = 'Previous turns in this session:\n' + '\n'.join(
            f"- {h['role'].upper()}: {h['content'][:500]}" for h in history[-6:]
        )

    user_prompt = build_user_prompt(
        task=request.task,
        external_context=ext_context_text,
        step_count=0,
        max_steps=max_steps,
        extracted_count=0,
        events_created=0,
        last_action='',
        last_result='',
        history_context=history_context,
        known_facts=state.known_facts,
        pre_parsed=pre_parsed,
    )
    messages.append(HumanMessage(content=user_prompt))

    # ── Agentic loop ────────────────────────────────────────────────────────
    while state.step_count < max_steps:
        state.step_count += 1

        # 1. LLM decision (isolated copy — retries never corrupt main list)
        decision, parse_errors = await _get_decision(list(messages), model)
        errors.extend(parse_errors)
        model_calls += 1

        if decision is None:
            return ArenaResponse(
                request_id=request.request_id,
                status='contract_error',
                final_response='Agent could not produce a valid decision after retries.',
                steps=state.step_count,
                stop_reason='contract_validation_failed',
                tool_calls=tool_traces,
                errors=errors,
                events=events_log,
                metrics=Metrics(model_calls=model_calls),
            )

        events_log.append({
            'step': state.step_count,
            'event': 'decision',
            'status': decision.status,
            'action': decision.action,
        })

        # ── Guard: completed+action is contradictory
        if decision.status == 'completed' and decision.action:
            log.warning('Step %d: completed+action — correcting to continue.', state.step_count)
            decision.status = 'continue'

        # ── Guard: needs_clarification when pre-parser already found the answer
        if decision.status == 'needs_clarification':
            missing = _check_truly_missing(decision, pre_parsed, state)
            if not missing:
                # Model is asking unnecessarily — nudge it with what we know
                log.warning('Step %d: spurious clarification — pre-parser has all facts.', state.step_count)
                hint = _build_clarification_override(pre_parsed, state, request.task)
                messages.append(HumanMessage(content=hint))
                state.step_count -= 1  # don't count this as a wasted step
                continue

        # ── Guard: completed with nothing done when context exists
        if (
            decision.status == 'completed'
            and not state.extracted_items
            and state.calendar_events_created == 0
            and (ext_context_text.strip() or _has_actionable_task(request.task, pre_parsed))
        ):
            log.warning('Step %d: premature completed — forcing extraction.', state.step_count)
            messages.append(HumanMessage(content=(
                'You returned "completed" but no action items were extracted and no calendar '
                'events were created. You MUST call extract_action_items first, then '
                'create_calendar_event for each item. Do NOT return completed yet.'
            )))
            continue

        # 2. Terminal statuses
        if decision.status == 'needs_clarification':
            msg = decision.user_message or 'Please provide more details (assignee and/or due date).'
            return ArenaResponse(
                request_id=request.request_id,
                status='needs_clarification',
                final_response=msg,
                steps=state.step_count,
                stop_reason='awaiting_user_input',
                tool_calls=tool_traces,
                errors=errors,
                events=events_log,
                metrics=Metrics(model_calls=model_calls),
            )

        if decision.status == 'completed':
            summary = _build_completion_summary(decision, state)
            return ArenaResponse(
                request_id=request.request_id,
                status='completed',
                final_response=summary,
                steps=state.step_count,
                stop_reason='goal_completed',
                tool_calls=tool_traces,
                errors=errors,
                events=events_log,
                metrics=Metrics(model_calls=model_calls),
            )

        if decision.status in ('blocked', 'failed'):
            msg = decision.user_message or 'Agent is blocked and cannot proceed.'
            return ArenaResponse(
                request_id=request.request_id,
                status=decision.status,
                final_response=msg,
                steps=state.step_count,
                stop_reason=decision.status,
                tool_calls=tool_traces,
                errors=errors,
                events=events_log,
                metrics=Metrics(model_calls=model_calls),
            )

        # 3. status == 'continue' → validate action
        if not decision.action:
            errors.append({
                'step': state.step_count,
                'type': 'no_action',
                'detail': 'Decision said continue but no action specified',
            })
            messages.append(HumanMessage(
                content='Error: You said continue but provided no action. Choose a valid action or stop.'
            ))
            continue

        if decision.action not in TOOL_REGISTRY:
            errors.append({
                'step': state.step_count,
                'type': 'rejected_action',
                'detail': f'Action {decision.action} not permitted',
            })
            tool_traces.append(ToolTrace(
                step=state.step_count, tool=decision.action,
                attempt=1, outcome='rejected', latency_ms=0,
            ))
            messages.append(HumanMessage(
                content=f'Error: Action "{decision.action}" is not permitted. Use only: {list(TOOL_REGISTRY)}'
            ))
            continue

        # Auto-fill arguments from pre-parser when model left them blank
        decision = _autofill_arguments(decision, pre_parsed, request.task)

        validation_error = _validate_arguments(decision)
        if validation_error:
            errors.append({
                'step': state.step_count,
                'type': 'semantic_validation',
                'detail': validation_error,
            })
            messages.append(HumanMessage(
                content=f'Validation error: {validation_error}. Fix and retry.'
            ))
            continue

        # 4. Execute tool with bounded retries
        from app.tools import execute_tool

        tool_result = None
        for attempt in range(1, settings.max_tool_retries + 2):
            tool_start = time.perf_counter()
            tool_result = await execute_tool(decision.action, decision.arguments, fault, attempt)
            tool_latency = (time.perf_counter() - tool_start) * 1000

            outcome = _classify_tool_outcome(tool_result)
            tool_traces.append(ToolTrace(
                step=state.step_count,
                tool=decision.action,
                attempt=attempt,
                outcome=outcome,
                latency_ms=round(tool_latency, 2),
            ))

            if outcome == 'success':
                break

            if attempt <= settings.max_tool_retries:
                log.info('Tool %s attempt %d failed (%s), retrying…', decision.action, attempt, outcome)
                continue

            errors.append({
                'step': state.step_count,
                'type': 'tool_failure',
                'detail': f'{decision.action} failed after {attempt} attempts: {outcome}',
            })
            return ArenaResponse(
                request_id=request.request_id,
                status='tool_error',
                final_response=f'Tool "{decision.action}" failed after {attempt} attempts. Stopping gracefully.',
                steps=state.step_count,
                stop_reason='tool_failure_exhausted',
                tool_calls=tool_traces,
                errors=errors,
                events=events_log,
                metrics=Metrics(model_calls=model_calls),
            )

        # 5. Update state and feed concise observation back
        state.last_action = decision.action
        state.last_result = json.dumps(tool_result, default=str)[:500]

        if decision.action_items:
            existing_descs = {i.description for i in state.extracted_items}
            for item in decision.action_items:
                if item.description not in existing_descs:
                    state.extracted_items.append(item)
                    existing_descs.add(item.description)
            state.known_facts = [
                f"{i.assignee or 'unknown'}: {i.description} by {i.due_date or 'TBD'}"
                for i in state.extracted_items
            ]

        if decision.action == 'create_calendar_event' and tool_result.get('status') == 'success':
            state.calendar_events_created += 1

        observation = _build_observation(decision.action, tool_result, state)
        messages.append(HumanMessage(content=observation))

        next_prompt = build_user_prompt(
            task=request.task,
            external_context=ext_context_text,
            step_count=state.step_count,
            max_steps=max_steps,
            extracted_count=len(state.extracted_items),
            events_created=state.calendar_events_created,
            last_action=state.last_action,
            last_result=state.last_result,
            history_context=history_context,
            known_facts=state.known_facts,
            pre_parsed=pre_parsed,
        )
        messages.append(HumanMessage(content=next_prompt))

    return ArenaResponse(
        request_id=request.request_id,
        status='budget_exceeded',
        final_response=f'Agent reached the maximum step limit ({max_steps}) without completing the task.',
        steps=state.step_count,
        stop_reason='step_budget_exceeded',
        tool_calls=tool_traces,
        errors=errors,
        events=events_log,
        metrics=Metrics(model_calls=model_calls),
    )


# ---------------------------------------------------------------------------
# LLM decision — isolated from main message list
# ---------------------------------------------------------------------------

async def _get_decision(
    messages: list,
    model: str,
) -> tuple[Optional[AgentDecision], list[dict]]:
    """
    Call LLM and validate output. Uses a local copy so retries never
    pollute the main conversation history.
    """
    llm = _get_llm(model)
    local_messages = list(messages)
    parse_errors: list[dict] = []

    for retry in range(settings.max_tool_retries + 1):
        try:
            response = llm.invoke(local_messages)
            raw_text = _extract_json(response.content)
            parsed = json.loads(raw_text)
            decision = AgentDecision.model_validate(parsed)
            return decision, parse_errors

        except (json.JSONDecodeError, ValidationError, Exception) as exc:
            detail = str(exc)[:300]
            parse_errors.append({
                'type': 'contract_parse_error',
                'detail': f'Attempt {retry + 1}: {detail}',
            })
            log.warning('Decision parse failed (attempt %d): %s', retry + 1, exc)

            if retry < settings.max_tool_retries:
                local_messages.append(HumanMessage(
                    content=(
                        f'Your previous output was invalid JSON or failed schema validation: {detail}. '
                        'Return ONLY valid JSON matching the AgentDecision schema. '
                        'No markdown, no text outside the JSON object.'
                    )
                ))

    return None, parse_errors


# ---------------------------------------------------------------------------
# Pre-parser helpers
# ---------------------------------------------------------------------------

def _check_truly_missing(
    decision: AgentDecision,
    pre_parsed: dict,
    state: AgentState,
) -> bool:
    """
    Return True only if something genuinely cannot be resolved.
    If pre_parsed already has date+assignee, nothing is missing.
    """
    has_date = bool(pre_parsed.get('date')) or bool(state.extracted_items)
    has_assignee = bool(pre_parsed.get('assignee')) or bool(state.extracted_items)
    return not (has_date and has_assignee)


def _has_actionable_task(task: str, pre_parsed: dict) -> bool:
    """Return True if the task string looks like a direct event creation request."""
    keywords = ['create', 'schedule', 'add', 'book', 'set up', 'calendar event']
    return (
        any(k in task.lower() for k in keywords)
        and bool(pre_parsed.get('date') or pre_parsed.get('assignee'))
    )


def _build_clarification_override(
    pre_parsed: dict,
    state: AgentState,
    task: str,
) -> str:
    """
    Build an override message that gives the model all the facts it was
    asking for, so it can proceed without another LLM round-trip.
    """
    parts = [
        'CORRECTION: You asked for clarification, but all required information is '
        'already present. Do NOT ask again. Here is what was found:\n'
    ]
    if pre_parsed.get('title'):
        parts.append(f'  • Title: {pre_parsed["title"]}')
    if pre_parsed.get('assignee'):
        parts.append(f'  • Assignee: {pre_parsed["assignee"]}')
    if pre_parsed.get('date'):
        parts.append(f'  • Date: {pre_parsed["date"]}')
    if state.extracted_items:
        parts.append(f'  • Extracted items: {len(state.extracted_items)} item(s) already in state')

    parts.append(
        '\nProceed immediately: call create_calendar_event with the above fields. '
        'Return status="continue" and action="create_calendar_event".'
    )
    return '\n'.join(parts)


def _autofill_arguments(
    decision: AgentDecision,
    pre_parsed: dict,
    task: str,
) -> AgentDecision:
    """
    When the model calls create_calendar_event but left fields empty,
    fill them from pre_parsed so validation passes.
    """
    if decision.action != 'create_calendar_event':
        return decision

    args = dict(decision.arguments)
    if not args.get('title') and pre_parsed.get('title'):
        args['title'] = pre_parsed['title']
    if not args.get('assignee') and pre_parsed.get('assignee'):
        args['assignee'] = pre_parsed['assignee']
    if not args.get('date') and pre_parsed.get('date'):
        args['date'] = pre_parsed['date']

    # If title is still missing, derive from task text
    if not args.get('title'):
        # Strip common boilerplate and use cleaned task as title
        cleaned = re.sub(
            r'\b(create|schedule|add|book|set up|a |an |the |calendar event|event)\b',
            '', task, flags=re.IGNORECASE,
        ).strip(' .,')
        if cleaned:
            args['title'] = cleaned[:80]

    decision.arguments = args
    return decision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_tool_outcome(tool_result: dict) -> str:
    status = tool_result.get('status', '')
    if status == 'timeout':
        return 'timeout'
    if status in ('error', 'exception'):
        return 'exception'
    if status == 'rejected':
        return 'rejected'
    if 'unexpected_field' in tool_result:
        return 'malformed_output'
    return 'success'


def _build_observation(action: str, tool_result: dict, state: AgentState) -> str:
    status = tool_result.get('status', 'unknown')
    data = tool_result.get('data', {})

    if action == 'extract_action_items':
        if status == 'success':
            count = len(state.extracted_items)
            return (
                f"Tool 'extract_action_items' succeeded. "
                f"{count} action item(s) now in state. "
                f"Next: call create_calendar_event for each item."
            )
        return f"Tool 'extract_action_items' failed: {tool_result.get('error', 'unknown')}."

    if action == 'create_calendar_event':
        if status == 'success':
            return (
                f"Tool 'create_calendar_event' succeeded. "
                f"Event '{data.get('title', '?')}' on {data.get('date', '?')} "
                f"for {data.get('assignee', 'unassigned')} (id={data.get('event_id', '?')}). "
                f"Total events: {state.calendar_events_created + 1}."
            )
        return f"Tool 'create_calendar_event' failed: {tool_result.get('error', 'unknown')}."

    if action == 'request_clarification':
        return f"Clarification sent: \"{data.get('question', '?')}\""

    return f"Tool '{action}' returned: {json.dumps(tool_result, default=str)[:400]}"


def _build_completion_summary(decision: AgentDecision, state: AgentState) -> str:
    if decision.user_message and len(decision.user_message) > 40:
        return decision.user_message
    parts = ['Meeting notes processed successfully.']
    if state.extracted_items:
        parts.append(f'{len(state.extracted_items)} action item(s) extracted:')
        for item in state.extracted_items:
            who = item.assignee or 'unassigned'
            when = item.due_date or 'no date'
            parts.append(f'  \u2022 {item.description} \u2014 {who} by {when}')
    if state.calendar_events_created:
        parts.append(f'{state.calendar_events_created} sandbox calendar event(s) created.')
    return '\n'.join(parts)


def _validate_arguments(decision: AgentDecision) -> Optional[str]:
    if decision.action == 'create_calendar_event':
        args = decision.arguments
        if not args.get('title'):
            return 'create_calendar_event requires a "title" argument.'
        date = args.get('date', '')
        if not date:
            return 'create_calendar_event requires a "date" argument.'
        if not _is_valid_date(date):
            return f'Invalid date format: "{date}". Use YYYY-MM-DD.'
    if decision.action == 'request_clarification':
        if not decision.arguments.get('question'):
            return 'request_clarification requires a "question" argument.'
    return None


def _is_valid_date(date_str: str) -> bool:
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return False
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


from app.tools import TOOL_REGISTRY  # noqa: E402
