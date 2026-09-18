/**
 * #14111 — Responses->Chat translation of tool-call outputs containing input_image
 * must keep the tool result textual and lift images into a following user message.
 *
 * Without this fix:
 * - `function_call_output` and `custom_tool_call_output` with an array output
 *   containing `input_image` parts get `JSON.stringify`'d into the `tool` message
 *   content, embedding the raw ~52KB base64 data URI as inert text.
 * - The model never receives the image (no structured `image_url` part).
 * - A single screenshot pushes ~50KB+ of meaningless base64 into the prompt.
 */
import test from "node:test";
import assert from "node:assert/strict";

const { openaiResponsesToOpenAIRequest } =
  await import("../../open-sse/translator/request/openai-responses.ts");

const IMAGE_PLACEHOLDER = "[Image attached in following user message]";
const SAMPLE_BASE64 = "AAAA" + "a".repeat(100); // small but realistic-looking base64

function findImageMessage(messages: Record<string, unknown>[]) {
  return messages.find(
    (message) =>
      message.role === "user" &&
      Array.isArray(message.content) &&
      message.content.some(
        (part) =>
          typeof part === "object" &&
          part !== null &&
          (part as Record<string, unknown>).type === "image_url"
      )
  );
}

test("#14111 function_call_output lifts input_image and preserves input_text", () => {
  const result = openaiResponsesToOpenAIRequest(
    "gpt-5.2",
    {
      input: [
        {
          type: "function_call",
          call_id: "call_abc123",
          name: "bash",
          arguments: '{"command":"ls"}',
        },
        {
          type: "function_call_output",
          call_id: "call_abc123",
          output: [
            { type: "input_text", text: "Script completed\nFile: screenshot.png" },
            {
              type: "input_image",
              image_url: `data:image/png;base64,${SAMPLE_BASE64}`,
              detail: "original",
            },
          ],
        },
      ],
    },
    false,
    {}
  );

  const messages = (result as Record<string, unknown>).messages as Record<string, unknown>[];
  // Should have: user message (placeholder) + function_call + tool result = 3 messages
  // Actually: instructions is empty, so no system message.
  // user placeholder (from input normalization) + assistant (from function_call) + tool result
  const toolMsg = messages.find((m) => m.role === "tool");
  assert.ok(toolMsg, "should have a tool message");
  assert.equal(typeof toolMsg.content, "string");
  assert.doesNotMatch(
    toolMsg.content as string,
    /base64|AAAA/,
    "tool content must not contain raw base64"
  );
  assert.ok(
    (toolMsg.content as string).includes("Script completed"),
    "text parts must be preserved"
  );
  assert.ok(
    (toolMsg.content as string).includes(IMAGE_PLACEHOLDER),
    "tool content must point to the following image message"
  );

  const imageMsg = findImageMessage(messages);
  assert.ok(imageMsg, "should lift the image into a user message");
  const imageContent = imageMsg.content as Record<string, unknown>[];
  assert.deepEqual(imageContent, [
    { type: "text", text: "Image output from tool call call_abc123:" },
    {
      type: "image_url",
      image_url: {
        url: `data:image/png;base64,${SAMPLE_BASE64}`,
        detail: "original",
      },
    },
  ]);
});

test("#14111 custom_tool_call_output lifts input_image and preserves input_text", () => {
  const result = openaiResponsesToOpenAIRequest(
    "gpt-5.2",
    {
      input: [
        {
          type: "custom_tool_call",
          call_id: "call_def456",
          name: "take_screenshot",
          input: "{}",
        },
        {
          type: "custom_tool_call_output",
          call_id: "call_def456",
          output: [
            { type: "input_text", text: "Screenshot captured" },
            {
              type: "input_image",
              image_url: `data:image/png;base64,${SAMPLE_BASE64}`,
              detail: "original",
            },
          ],
        },
      ],
    },
    false,
    {}
  );

  const messages = (result as Record<string, unknown>).messages as Record<string, unknown>[];
  const toolMsg = messages.find((m) => m.role === "tool");
  assert.ok(toolMsg, "should have a tool message");
  assert.equal(typeof toolMsg.content, "string");
  assert.doesNotMatch(
    toolMsg.content as string,
    /base64|AAAA/,
    "tool content must not contain raw base64"
  );
  assert.ok(
    (toolMsg.content as string).includes("Screenshot captured"),
    "text parts must be preserved"
  );
  assert.ok(
    (toolMsg.content as string).includes(IMAGE_PLACEHOLDER),
    "tool content must point to the following image message"
  );

  const imageMsg = findImageMessage(messages);
  assert.ok(imageMsg, "should lift the image into a user message");
  const imageContent = imageMsg.content as Record<string, unknown>[];
  assert.deepEqual(imageContent, [
    { type: "text", text: "Image output from tool call call_def456:" },
    {
      type: "image_url",
      image_url: {
        url: `data:image/png;base64,${SAMPLE_BASE64}`,
        detail: "original",
      },
    },
  ]);
});

test("#14111 parallel tool images stay after every tool result", () => {
  const result = openaiResponsesToOpenAIRequest(
    "gpt-5.2",
    {
      input: [
        {
          type: "function_call",
          call_id: "call_one",
          name: "capture_one",
          arguments: "{}",
        },
        {
          type: "function_call",
          call_id: "call_two",
          name: "capture_two",
          arguments: "{}",
        },
        {
          type: "function_call_output",
          call_id: "call_one",
          output: [{ type: "input_image", image_url: "data:image/png;base64,ONE" }],
        },
        {
          type: "function_call_output",
          call_id: "call_two",
          output: [{ type: "input_image", image_url: "data:image/png;base64,TWO" }],
        },
      ],
    },
    false,
    {}
  );

  const messages = (result as Record<string, unknown>).messages as Record<string, unknown>[];
  assert.deepEqual(
    messages.map((message) => message.role),
    ["assistant", "tool", "tool", "user", "user"],
    "all tool results must remain contiguous before lifted image messages"
  );
});

test("#8459 string output is unchanged", () => {
  const result = openaiResponsesToOpenAIRequest(
    "gpt-5.2",
    {
      input: [
        {
          type: "function_call",
          call_id: "call_ghi789",
          name: "grep",
          arguments: '{"pattern":"foo"}',
        },
        {
          type: "function_call_output",
          call_id: "call_ghi789",
          output: "Found 3 matches",
        },
      ],
    },
    false,
    {}
  );

  const messages = (result as Record<string, unknown>).messages as Record<string, unknown>[];
  const toolMsg = messages.find((m) => m.role === "tool");
  assert.ok(toolMsg, "should have a tool message");
  assert.equal(toolMsg.content, "Found 3 matches", "string output must pass through unchanged");
});

test("#8459 JSON object output is unchanged (not an array of content parts)", () => {
  const result = openaiResponsesToOpenAIRequest(
    "gpt-5.2",
    {
      input: [
        {
          type: "function_call",
          call_id: "call_jkl012",
          name: "read_file",
          arguments: '{"path":"file.txt"}',
        },
        {
          type: "function_call_output",
          call_id: "call_jkl012",
          output: { result: "file content", metadata: { size: 123 } },
        },
      ],
    },
    false,
    {}
  );

  const messages = (result as Record<string, unknown>).messages as Record<string, unknown>[];
  const toolMsg = messages.find((m) => m.role === "tool");
  assert.ok(toolMsg, "should have a tool message");
  // JSON object should still be stringified, but NOT an array of content parts
  assert.equal(typeof toolMsg.content, "string");
  assert.ok(
    (toolMsg.content as string).includes("file content"),
    "JSON output should be stringified"
  );
  // No image placeholder for non-content-part arrays
  assert.doesNotMatch(
    toolMsg.content as string,
    /\[Image omitted/,
    "non-content-part array must not be treated as image-bearing"
  );
});
