package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	p "github.com/anishathalye/porcupine"
)

type rec struct {
	Proc int    `json:"proc"`
	Op   uint32 `json:"op"`   // 0=enq, 1=deq
	Arg  uint64 `json:"arg"`  // enq value
	RetV uint64 `json:"ret"`  // enq: 1=success, 0=failure ; deq: value or 0 for empty
	Call int64  `json:"call"`
	End  int64  `json:"end"`
}

type input struct {
	Kind string
	Val  uint64
}

type output struct {
	Ok    bool
	Empty bool
	Val   uint64
}

func main() {
	if len(os.Args) != 2 && len(os.Args) != 3 {
		log.Fatalf("usage: %s <history.jsonl> [out.html]", os.Args[0])
	}

	inPath := os.Args[1]
	var outPath string
	if len(os.Args) == 3 {
		outPath = os.Args[2]
	}

	f, err := os.Open(inPath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	var ops []p.Operation
	s := bufio.NewScanner(f)

	for s.Scan() {
		var r rec
		if err := json.Unmarshal(s.Bytes(), &r); err != nil {
			log.Fatal(err)
		}

		var in input
		var out output

		if r.Op == 0 {
			// enqueue
			in = input{Kind: "Enq", Val: r.Arg}
			out = output{Ok: r.RetV != 0}
		} else {
			// dequeue
			in = input{Kind: "Deq"}
			if r.RetV == 0 {
				out = output{Ok: false, Empty: true}
			} else {
				out = output{Ok: true, Empty: false, Val: r.RetV}
			}
		}

		ops = append(ops, p.Operation{
			ClientId: r.Proc,
			Input:    in,
			Output:   out,
			Call:     r.Call,
			Return:   r.End,
		})
	}

	if err := s.Err(); err != nil {
		log.Fatal(err)
	}

	model := p.Model{
		Init: func() interface{} { return []uint64{} },

		Step: func(state interface{}, in, out interface{}) (bool, interface{}) {
			q := append([]uint64(nil), state.([]uint64)...)
			i := in.(input)
			o := out.(output)

			switch i.Kind {
			case "Enq":
				// failed enqueue = legal no-op
				if !o.Ok {
					return true, q
				}
				return true, append(q, i.Val)

			case "Deq":
				if len(q) == 0 {
					return o.Empty, q
				}
				if o.Empty {
					return false, state
				}
				if o.Val != q[0] {
					return false, state
				}
				return true, q[1:]

			default:
				return false, state
			}
		},

		Equal: func(a, b interface{}) bool {
			xa := a.([]uint64)
			xb := b.([]uint64)
			if len(xa) != len(xb) {
				return false
			}
			for i := range xa {
				if xa[i] != xb[i] {
					return false
				}
			}
			return true
		},
	}

	timeout := 10 * time.Second
	res, info := p.CheckOperationsVerbose(model, ops, timeout)

	if res == p.Ok {
		fmt.Println("Linearizable (FIFO)")
	} else if res == p.Illegal {
		fmt.Println("Not linearizable (FIFO)")
	} else {
		fmt.Println("Unknown (timeout)")
	}

	if outPath != "" {
		if err := p.VisualizePath(model, info, outPath); err != nil {
			log.Fatal(err)
		}
		fmt.Println("wrote", outPath)
	}

	if res == p.Ok {
		os.Exit(0)
	}
	os.Exit(1)
}