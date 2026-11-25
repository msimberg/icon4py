#!/usr/bin/env python3
"""Restrict the review sandbox to DNS and an HTTPS host allowlist.

Usage (run as root with the system python3; stdlib only):
    sudo python3 netns_firewall.py setup <host> [<host>...]
    sudo python3 netns_firewall.py teardown

A veth pair connects the namespace to the host. The namespace permits DNS to
the host resolvers and TCP port 443 to the allowlisted IP addresses. The
addresses are resolved during setup, so DNS responses cannot bypass the
allowlist. The review driver drops from root to the calling user on entry.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys

NETNS = "weekly-review"
VETH_HOST = "v-wr-host"
VETH_NS = "v-wr-ns"
SUBNET = "10.201.0.0/30"
HOST_IP = "10.201.0.1"
NS_IP = "10.201.0.2"
NETNS_DIR = f"/etc/netns/{NETNS}"


def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def sh(cmd: list[str], *, input: str | None = None) -> str:
    result = subprocess.run(cmd, input=input, capture_output=True, text=True)
    if result.returncode != 0:
        die(f"command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout


def require_root() -> None:
    if os.geteuid() != 0:
        die("this command must run as root (use sudo)")


def resolve_ips(hosts: list[str]) -> list[str]:
    ips: set[str] = set()
    for host in hosts:
        try:
            infos = socket.getaddrinfo(host, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        except socket.gaierror:
            die(f"could not resolve host: {host}")
        ips.update(info[4][0] for info in infos)
    return sorted(ips)


def resolvers() -> list[str]:
    servers = []
    with open("/etc/resolv.conf", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "nameserver":
                servers.append(parts[1])
    return sorted(set(servers))


def default_iface() -> str:
    tokens = sh(["ip", "route", "get", "1.1.1.1"]).split()
    if "dev" in tokens:
        return tokens[tokens.index("dev") + 1]
    die("cannot determine the default network interface")

def ns_exists() -> bool:
    return NETNS in sh(["ip", "netns", "list"])


def nft_ruleset(allowed: list[str], resolv: list[str]) -> str:
    ip_list = ", ".join(allowed) or "0.0.0.0"
    dns_list = ", ".join(resolv) or "0.0.0.0"
    return f"""flush ruleset
table inet weekly-review-filter {{
    chain input {{
        type filter hook input priority 0; policy drop;
        iifname "lo" accept
        ct state established,related accept
    }}
    chain output {{
        type filter hook output priority 0; policy drop;
        oifname "lo" accept
        ct state established,related accept
        ip daddr {{ {dns_list} }} udp dport 53 accept
        ip daddr {{ {dns_list} }} tcp dport 53 accept
        ip daddr {{ {ip_list} }} tcp dport 443 accept
    }}
}}
"""


def cmd_setup(hosts: list[str]) -> None:
    require_root()
    if ns_exists():
        die(f"namespace {NETNS} already exists (run teardown first)")
    if not hosts:
        die("usage: netns_firewall.py setup <host> [<host>...]")

    resolved = [(host, resolve_ips([host])) for host in hosts]
    allowed = sorted(
        {
            address
            for _, addresses in resolved
            for address in addresses
            if not address.startswith(("127.", "10.201."))
        }
    )
    if not allowed:
        die(f"no usable addresses for: {hosts}")
    print(f"Allowed egress IPs: {', '.join(allowed)}")

    wan = default_iface()
    sh(["ip", "netns", "add", NETNS])
    try:
        sh(["ip", "link", "add", VETH_HOST, "type", "veth", "peer", "name", VETH_NS])
        sh(["ip", "link", "set", VETH_NS, "netns", NETNS])
        sh(["ip", "addr", "add", f"{HOST_IP}/30", "dev", VETH_HOST])
        sh(["ip", "link", "set", VETH_HOST, "up"])
        sh(["ip", "-n", NETNS, "addr", "add", f"{NS_IP}/30", "dev", VETH_NS])
        sh(["ip", "-n", NETNS, "link", "set", VETH_NS, "up"])
        sh(["ip", "-n", NETNS, "link", "set", "lo", "up"])
        sh(["ip", "-n", NETNS, "route", "add", "default", "via", HOST_IP])

        # Per-namespace resolv.conf (ip netns exec bind-mounts it automatically).
        os.makedirs(NETNS_DIR, exist_ok=True)
        with open(f"{NETNS_DIR}/resolv.conf", "w", encoding="utf-8") as handle:
            for server in resolvers():
                handle.write(f"nameserver {server}\n")

        # ip netns exec bind-mounts this over /etc/hosts, so required hosts do
        # not depend on a resolver inside the namespace. URLs retain their
        # original names for TLS validation.
        with open(f"{NETNS_DIR}/hosts", "w", encoding="utf-8") as handle:
            for host, addresses in resolved:
                for address in addresses:
                    handle.write(f"{address}\t{host}\n")

        # NAT the subnet out of the host's default interface and allow forwarding.
        sh(["iptables", "-t", "nat", "-A", "POSTROUTING", "-s", SUBNET, "-o", wan, "-j", "MASQUERADE"])
        sh(["iptables", "-A", "FORWARD", "-s", SUBNET, "-j", "ACCEPT"])
        sh(["iptables", "-A", "FORWARD", "-d", SUBNET, "-m", "state",
            "--state", "ESTABLISHED,RELATED", "-j", "ACCEPT"])

        # Lock down output inside the namespace.
        sh(["ip", "netns", "exec", NETNS, "nft", "-f", "-"], input=nft_ruleset(allowed, resolvers()))
    except SystemExit:
        # Roll the namespace back so a failed setup does not leave junk behind.
        subprocess.run(["ip", "netns", "del", NETNS], capture_output=True)
        raise

    print(f"Namespace {NETNS} ready.")


def cmd_teardown() -> None:
    require_root()
    wan = default_iface()
    if ns_exists():
        sh(["ip", "netns", "del", NETNS])
    subprocess.run(
        ["iptables", "-t", "nat", "-D", "POSTROUTING", "-s", SUBNET, "-o", wan, "-j", "MASQUERADE"],
        capture_output=True,
    )
    subprocess.run(["iptables", "-D", "FORWARD", "-s", SUBNET, "-j", "ACCEPT"], capture_output=True)
    subprocess.run(
        ["iptables", "-D", "FORWARD", "-d", SUBNET, "-m", "state",
         "--state", "ESTABLISHED,RELATED", "-j", "ACCEPT"],
        capture_output=True,
    )
    subprocess.run(["rm", "-rf", NETNS_DIR], check=False)
    print(f"Namespace {NETNS} removed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Netns firewall for the weekly review sandbox.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_setup = sub.add_parser("setup", help="Create the namespace and firewall rules.")
    p_setup.add_argument("hosts", nargs="+", help="Hosts the sandbox may reach over HTTPS.")
    sub.add_parser("teardown", help="Remove the namespace and firewall rules.")
    args = parser.parse_args()
    if args.command == "setup":
        cmd_setup(args.hosts)
    else:
        cmd_teardown()


if __name__ == "__main__":
    main()
