import re

HEADER_RE = re.compile(rb"\x13..\x00\x00.\x00{7}", re.DOTALL)

known_header = bytes.fromhex("1300200000dd00000000000000f01f00")
print("known_header hex:", known_header.hex())
print("known_header len:", len(known_header))

m = HEADER_RE.search(known_header)
print("direct search on isolated bytes:", m)
if m:
    print("matched span:", m.span(), "matched bytes:", m.group().hex())

# now embed it in a larger buffer at a specific offset, like the real case
buf = b"\x00" * 100 + known_header + b"\x00" * 100
m2 = HEADER_RE.search(buf, 50, 50 + 8300)
print("search with pos/endpos in larger buffer:", m2)
if m2:
    print("start:", m2.start(), "expected around:", 100)
