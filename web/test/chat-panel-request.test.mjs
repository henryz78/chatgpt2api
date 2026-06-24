import assert from "node:assert/strict";
import test from "node:test";

import { buildChatCompletionRequest, buildChatCompletionRequestBody } from "../src/app/debug/components/chat-panel-request.ts";

const messages = [{ role: "user", content: "think carefully" }];

test("adds selected thinking_effort to chat completions body", () => {
  assert.deepEqual(
    buildChatCompletionRequestBody({
      model: " auto ",
      messages,
      effortField: "thinking_effort",
      effortLevel: "high",
    }),
    {
      model: "auto",
      messages,
      thinking_effort: "high",
    },
  );
});

test("targets chat completions with selected reasoning_effort", () => {
  assert.deepEqual(
    buildChatCompletionRequest({
      model: "auto",
      messages,
      effortField: "reasoning_effort",
      effortLevel: "medium",
    }),
    {
      path: "/v1/chat/completions",
      options: {
        method: "POST",
        body: {
          model: "auto",
          messages,
          reasoning_effort: "medium",
        },
      },
    },
  );
});

test("adds selected reasoning_effort to chat completions body", () => {
  assert.deepEqual(
    buildChatCompletionRequestBody({
      model: "",
      messages,
      effortField: "reasoning_effort",
      effortLevel: "xhigh",
    }),
    {
      model: "auto",
      messages,
      reasoning_effort: "xhigh",
    },
  );
});

test("omits effort when level is empty", () => {
  assert.deepEqual(
    buildChatCompletionRequestBody({
      model: "gpt-5",
      messages,
      effortField: "thinking_effort",
      effortLevel: "",
    }),
    {
      model: "gpt-5",
      messages,
    },
  );
});
