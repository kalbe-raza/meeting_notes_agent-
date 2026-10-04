# Submission Summary

Full name: Mubariz
Roll number: i220794
Class/ section: B
GitHub username: kalbe-raza
Agent name: Meeting Notes Agent
Domain: Meeting Notes Agent (Domain 6)

GitHub repository URL: https://github.com/kalbe-raza/meeting_notes_agent-
Working agent interface: https://meeting-notes-agent-59bo.onrender.com
Health endpoint (GET): https://meeting-notes-agent-59bo.onrender.com/healt
Arena endpoint (POST): https://meeting-notes-agent-59bo.onrender.com/arena/run
Manifest endpoint (GET): https://meeting-notes-agent-59bo.onrender.com/arena/manifest
API documentation: https://meeting-notes-agent-59bo.onrender.com/docs

Hosting provider: Render
Default model/ provider: labs-leanstral-1-5 via Mistral AI
Other available models: None configured
Example input: "Extract action items: Ali will write the report on 2026-10-05."
Expected result: Status 'completed', mock calendar event created in sandbox.
Cold-start/ restart limitations: Render free tier sleeps after 15 minutes of inactivity; first request may take 30-50 seconds to wake up. Chat memory is stored in-RAM and resets on dyno restart (single worker configuration).
Repository access: Instructor invited (pending/accepted)
Public test results: tests/test_agent.py (All 9 tests passing, including 6 Reliability Arena categories)
