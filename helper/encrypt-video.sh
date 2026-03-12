#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input_file="${1:-$script_dir/../example_data/papier-abfall-mit-voice.mp4}"
public_key_path="${2:-$script_dir/../public_key.pem}"
output_file="${input_file}.enc"
session_key_file="${output_file}.session_key.b64"

[[ -f "$input_file" ]] || { echo "Input file not found: $input_file" >&2; exit 1; }
[[ -f "$public_key_path" ]] || { echo "Public key not found: $public_key_path" >&2; exit 1; }
command -v openssl >/dev/null 2>&1 || { echo "openssl not found in PATH" >&2; exit 1; }

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

openssl rand 32 > "$tmp_dir/key.bin"
openssl rand 16 > "$tmp_dir/iv.bin"

key_hex="$(od -An -tx1 -v "$tmp_dir/key.bin" | tr -d ' \n')"
iv_hex="$(od -An -tx1 -v "$tmp_dir/iv.bin" | tr -d ' \n')"

openssl enc -aes-256-cbc -e -nosalt \
  -K "$key_hex" \
  -iv "$iv_hex" \
  -in "$input_file" \
  -out "$tmp_dir/cipher.bin"

cat "$tmp_dir/iv.bin" "$tmp_dir/cipher.bin" > "$output_file"

openssl pkeyutl -encrypt \
  -pubin \
  -inkey "$public_key_path" \
  -in "$tmp_dir/key.bin" \
  -out "$tmp_dir/key.enc" \
  -pkeyopt rsa_padding_mode:oaep \
  -pkeyopt rsa_oaep_md:sha256 \
  -pkeyopt rsa_mgf1_md:sha256

session_key_b64="$(openssl base64 -A -in "$tmp_dir/key.enc")"
printf '%s\n' "$session_key_b64" > "$session_key_file"

echo "Encrypted Session Key (Base64):"
echo "$session_key_b64"
echo "Encrypted video written to: $output_file"
echo "Encrypted session key written to: $session_key_file"
