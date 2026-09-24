"""Meeting Notes Agent — bounded agentic loop with contract-gated execution."""

import json
import logging
import time
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
from app.prompts import SYSTEM_PROMPT, build_user_prompt, CLARIFICATION_FOLLOWUP_TEMPLATE
from app.tools import TOOL_REGISTRY, execute_tool

log = logging.getLogger('agent')

# Lazy LLM initialization
_llm = None


def _get_llm(model: str = ''):
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        # We use ChatOpenAI because OpenRouter is OpenAI-compatible
        _llm = ChatOpenAI(
            model=model or settings.model_name,
            temperature=0,
            max_tokens=settings.max_output_tokens,
            api_key=settings.openrouter_api_key,  # Your OpenRouter Key
            base_url=settings.llm_base_url,       # Points to OpenRouter, NOT OpenAI
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "MeetingNotesAgent"
            }
        )
    return _llm

async def run_agent(
    request: ArenaRequest,
    history: list[dict],
    model: str = 'unconfigured',
) -> ArenaResponse:
    """
    Main agentic loop. Called by arena.execute().
    Handles: LLM decisions → validation → tool execution → feedback → stop.
    """
    state = AgentState(goal=request.task)
    fault: Fault = request.arena_config.fault
    max_steps = request.arena_config.max_steps
    tool_traces: list[ToolTrace] = []
    errors: list[dict] = []
    events_log: list[dict] = []
    model_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # Format external context as untrusted data block
    ext_context_text = '\n'.join(
        f'[SOURCE: {ec.source} | TRUST: untrusted]\n{ec.content}'
        for ec in request.external_context
    )

    # Handle fault: invalid_agent_decision (simulate LLM returning garbage)
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

    # Build initial messages
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Add conversation history for multi-turn
    if history:
        history_context = '\n'.join(
            f"[{h['role']}]: {h['content'][:500]}" for h in history[-6:]
        )
    else:
        history_context = ''

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
    )
    messages.append(HumanMessage(content=user_prompt))

    # --- AGENTIC LOOP ---
    while state.step_count < max_steps:
        state.step_count += 1

        # 1. Get LLM decision
        decision = await _get_decision(messages, model, errors)
        model_calls += 1

        if decision is None:
            # Contract error — recovery failed
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

        # 2. Handle terminal statuses
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
            msg = decision.user_message or 'Meeting notes processed. Action items extracted and scheduled.'
            return ArenaResponse(
                request_id=request.request_id,
                status='completed',
                final_response=msg,
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

        # 3. status == 'continue' → validate and execute action
        if not decision.action:
            errors.append({'step': state.step_count, 'type': 'no_action', 'detail': 'Decision said continue but no action specified'})
            messages.append(HumanMessage(content='Error: You said continue but provided no action. Choose a valid action or stop.'))
            continue

        # Autonomy boundary check
        if decision.action not in TOOL_REGISTRY:
            errors.append({'step': state.step_count, 'type': 'rejected_action', 'detail': f'Action {decision.action} not permitted'})
            tool_traces.append(ToolTrace(
                step=state.step_count, tool=decision.action,
                attempt=1, outcome='rejected', latency_ms=0,
            ))
            messages.append(HumanMessage(content=f'Error: Action "{decision.action}" is not permitted. Use only: {list(TOOL_REGISTRY)}'))
            continue

        # Semantic validation of arguments
        validation_error = _validate_arguments(decision)
        if validation_error:
            errors.append({'step': state.step_count, 'type': 'semantic_validation', 'detail': validation_error})
            messages.append(HumanMessage(content=f'Validation error: {validation_error}. Fix and retry.'))
            continue

        # 4. Execute tool with retry
        tool_result = None
        for attempt in range(1, settings.max_tool_retries + 2):  # max 3 attempts
            tool_start = time.perf_counter()
            tool_result = await execute_tool(decision.action, decision.arguments, fault, attempt)
            tool_latency = (time.perf_counter() - tool_start) * 1000

            outcome = 'success'
            if tool_result.get('status') == 'timeout':
                outcome = 'timeout'
            elif tool_result.get('status') == 'error':
                outcome = 'exception'
            elif tool_result.get('status') == 'exception':
                outcome = 'exception'
            elif tool_result.get('status') == 'rejected':
                outcome = 'rejected'
            elif 'unexpected_field' in tool_result:
                outcome = 'malformed_output'

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
                log.info('Tool %s attempt %d failed (%s), retrying', decision.action, attempt, outcome)
                continue
            else:
                # All retries exhausted
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

        # 5. Feed result back to agent
        state.last_action = decision.action
        state.last_result = json.dumps(tool_result, default=str)[:500]
        state.extracted_items = decision.action_items if decision.action_items else state.extracted_items

        if decision.action == 'create_calendar_event' and tool_result.get('status') == 'success':
            state.calendar_events_created += 1

        # Observation feedback
        observation = f"Tool '{decision.action}' returned: {json.dumps(tool_result, default=str)[:800]}"
        messages.append(HumanMessage(content=observation))

        # Update prompt context for next iteration
        next_prompt = build_user_prompt(
            task=request.task,
            external_context=ext_context_text,
            step_count=state.step_count,
            max_steps=max_steps,
            extracted_count=len(state.extracted_items),
            events_created=state.calendar_events_created,
            last_action=state.last_action,
            last_result=state.last_result,
        )
        messages.append(HumanMessage(content=next_prompt))

    # --- Budget exceeded ---
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


async def _get_decision(messages: list, model: str, errors: list) -> Optional[AgentDecision]:
    """Call LLM and validate output against AgentDecision schema. Bounded retries."""
    llm = _get_llm(model)

    for retry in range(settings.max_tool_retries + 1):  # max 3 attempts
        try:
            response = llm.invoke(messages)
            raw_text = response.content.strip()

            # Strip markdown code fences if present
            if raw_text.startswith('```'):
                lines = raw_text.split('\n')
                raw_text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

            parsed = json.loads(raw_text)
            decision = AgentDecision.model_validate(parsed)
            return decision

        except (json.JSONDecodeError, ValidationError, Exception) as e:
            errors.append({
                'type': 'contract_parse_error',
                'detail': f'Attempt {retry + 1}: {str(e)[:200]}',
            })
            log.warning('Decision parse failed (attempt %d): %s', retry + 1, e)

            if retry < settings.max_tool_retries:
                messages.append(HumanMessage(
                    content=f'Your previous output was invalid JSON or failed schema validation: {str(e)[:200]}. '
                            f'Return ONLY valid JSON matching the AgentDecision schema.'
                ))

    return None


def _validate_arguments(decision: AgentDecision) -> Optional[str]:
    """Semantic validation of action arguments."""
    if decision.action == 'create_calendar_event':
        args = decision.arguments
        date = args.get('date', '')
        if date and not _is_valid_date(date):
            return f'Invalid date format: "{date}". Use YYYY-MM-DD.'
        if not args.get('title'):
            return 'create_calendar_event requires a "title" argument.'
        if not args.get('date'):
            return 'create_calendar_event requires a "date" argument.'

    if decision.action == 'request_clarification':
        if not decision.arguments.get('question'):
            return 'request_clarification requires a "question" argument.'

    return None


def _is_valid_date(date_str: str) -> bool:
    """Basic YYYY-MM-DD validation."""
    import re
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return False
    try:
        from datetime import datetime
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False
